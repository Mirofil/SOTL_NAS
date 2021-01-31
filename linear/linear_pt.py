import torch
from torch import Tensor
from torch.nn import Linear, MSELoss, functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import math
import itertools
from utils import evidence_log, evidence, c_tilde, sample_tau, sample_C, c_cov, sample_w, featurize, eval_features, data_generator, jacobian, hessian
from log_utils import AverageMeter, wandb_auth
from datasets import get_datasets
import wandb


class Linear2(nn.Linear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alphas = torch.nn.Parameter(torch.ones(1, self.in_features))
        # self.alphas = torch.nn.Parameter(torch.tensor([-0.0499, -0.0443,  0.1992]))
        # weights from a training run with D=3 - tensor([[-0.0499, -0.0443,  0.1992]]

    def forward(self, input: Tensor, weight: Tensor = None, alphas: Tensor = None) -> Tensor:
        if weight is None:
            weight = self.weight
        if alphas is None:
            alphas = self.alphas
        return F.linear(input, weight*F.softmax(alphas), self.bias)


# define the model
class Net(torch.nn.Module):
    def __init__(self, num_features = 2):
        super(Net, self).__init__()
        self.fc1 = Linear(num_features, 1, bias=False)

    def forward(self, x):
        return self.fc1(x)
    
    def weight_params(self):
        for p in [self.fc1.weight]:
            yield p
    
    def arch_params(self):
        for p in [self.fc1.alphas]:
            yield p

class SoTLNet(Net):
    def __init__(self, num_features = 2):
        super().__init__()
        self.fc1 = Linear2(num_features, 1, bias=False)

    def forward(self, x, weight=None, alphas=None):
        return self.fc1(x, weight, alphas)


def sotl_gradient(model, criterion, xs, ys, weight_buffer, order="first_order", hvp = "exact"):
    total_arch_gradient = 0
    if hvp == "exact":
        for i in range(1, min(T, len(weight_buffer))):
            loss = criterion(model(xs[i], weight_buffer[i][0], model.fc1.alphas[0]), ys[i])
            da = torch.autograd.grad(loss, model.arch_params(), retain_graph=True)[0]
            dw = torch.autograd.grad(loss, weight_buffer[i][0], retain_graph=True)[0]

            loss2 = criterion(model(xs[i], weight_buffer[i-1][0], model.fc1.alphas[0]), ys[i])
            hessian_matrix = hessian(loss2*1, weight_buffer[i-1][0], model.fc1.alphas).reshape(model.fc1.weight.size()[1],model.fc1.alphas.size()[1]) # TODO this whole line is WEIRD

            second_order_term = torch.matmul(dw, hessian_matrix)

            total_arch_gradient += da + (-w_lr*second_order_term)
            total_arch_gradient = [total_arch_gradient]
    elif hvp == "finite_diff":
        for i in range(1, min(T, len(weight_buffer))):
            loss = criterion(model(xs[i], weight_buffer[i][0], model.fc1.alphas[0]), ys[i])
            da = torch.autograd.grad(loss, model.arch_params(), retain_graph=True)[0]
            dw = torch.autograd.grad(loss, weight_buffer[i][0], retain_graph=True)[0]

            # Footnotes suggest to divide by L2 norm of the gradient
            norm = torch.cat([w.view(-1) for w in dw]).norm()
            eps = 0.01 / norm

            # w+ = w + eps*dw`
            with torch.no_grad():
                for p, d in zip(weight_buffer[i-1][0], dw):
                    p += eps * d

            loss2 = criterion(model(xs[i], weight_buffer[i-1][0], model.fc1.alphas[0]), ys[i])
            dalpha_pos = torch.autograd.grad(
                loss2, model.arch_params()
            )  # dalpha { L_trn(w+) }

            # w- = w - eps*dw`
            with torch.no_grad():
                for p, d in zip(weight_buffer[i-1][0], dw):
                    p -= 2.0 * eps * d
            loss3 = criterion(model(xs[i], weight_buffer[i-1][0], model.fc1.alphas[0]), ys[i])
            dalpha_neg = torch.autograd.grad(
                loss3, model.arch_params()
            )  # dalpha { L_trn(w-) }

            # recover w
            with torch.no_grad():
                for p, d in zip(weight_buffer[i-1][0], dw):
                    p += eps * d

            total_arch_gradient = [-w_lr*(p - n) / 2.0 * eps for p, n in zip(dalpha_pos, dalpha_neg)]
    

    return total_arch_gradient

def train_bptt(num_epochs, model, dset_train, batch_size, T, grad_clip, logging_freq):
    model.train()
    train_loader = torch.utils.data.DataLoader(dset_train, batch_size = batch_size*T, shuffle=True)

    for epoch in range(num_epochs):
        epoch_loss = AverageMeter()
        true_batch_index = 0
        for batch_idx, batch in enumerate(train_loader):
            xs, ys = torch.split(batch[0], batch_size), torch.split(batch[1], batch_size)

            weight_buffer = []
            for x, y in zip(xs,ys):
            
                weight_buffer.append([w.clone() for w in model.weight_params()])
                y_pred = model(x)
                loss = criterion(y_pred, y)
                epoch_loss.update(loss.item())

                grads = torch.autograd.grad(loss, model.weight_params(), retain_graph=True, allow_unused=True, create_graph=True)
                
                w_optimizer.zero_grad()

                with torch.no_grad():
                    for g, w in zip(grads, model.weight_params()):
                        w.grad = g
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                w_optimizer.step()
                true_batch_index += 1
                if true_batch_index % logging_freq == 0:
                    print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, true_batch_index, epoch_loss.avg))
                    wandb.log({"Train loss": epoch_loss.avg})

            total_arch_gradient = sotl_gradient(model, criterion, xs, ys, weight_buffer, hvp="finite_diff")

            a_optimizer.zero_grad()

            for g, w in zip(total_arch_gradient, model.arch_params()):
                w.grad = g
            torch.nn.utils.clip_grad_norm_(model.arch_params(), 1)
            a_optimizer.step()


def valid_func(model, val_loader, criterion):
    model.eval()
    val_meter = AverageMeter()
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            y_pred = model(x)
            val_loss = criterion(y_pred, y)
            val_meter.update(val_loss.item())
    print("Val loss: {}".format(val_meter.avg))
    return val_meter

def train_normal(num_epochs, model, dset_train, batch_size, grad_clip, logging_freq, optim="newton"):
    train_loader = torch.utils.data.DataLoader(dset_train, batch_size = batch_size, shuffle=True)

    model.train()
    for epoch in range(num_epochs):

        epoch_loss = AverageMeter()
        for batch_idx, batch in enumerate(train_loader):
            x, y = batch
            w_optimizer.zero_grad()


            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward(retain_graph=True)

            epoch_loss.update(loss.item())
            if optim == "newton":
                linear_weight = list(model.weight_params())[0]
                hessian_newton = torch.inverse(hessian(loss*1, linear_weight, linear_weight).reshape(linear_weight.size()[1], linear_weight.size()[1]))
                with torch.no_grad():
                    for w in model.weight_params():
                        w = w.subtract_(torch.matmul(w.grad, hessian_newton))
            else:
                torch.nn.utils.clip_grad_norm_(model.weight_params(), 1)
                w_optimizer.step()
            if batch_idx % logging_freq == 0:
                print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch_idx, epoch_loss.avg))
                wandb.log({"Train loss": epoch_loss.avg})

# # test the model
# model.eval()
# test_data = data_generator(1,max_order=D, featurize_type="polynomial")
# prediction = model(Variable(Tensor(test_data[0][0])))
# print("Prediction: {}".format(prediction.data[0]))
# print("Expected: {}".format(test_data[1][0]))


if __name__ == "__main__":
    
    wandb_auth()
    wandb.init(project="NAS", group=f"Linear_SOTL")

    ### PARAMS
    num_epochs = 500
    batch_size = 64
    D=torch.tensor(3)
    N=torch.tensor(50000)
    w_lr = 0.0001
    a_lr = 0.0001
    T = 10
    grad_clip = 1
    logging_freq=200

    ### MODEL INIT

    # x, y = data_generator(N, max_order=D, noise_var=1, featurize_type='polynomial')
    x, y = get_datasets("songs")
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    dset = torch.utils.data.TensorDataset(x, y)

    dset_train, dset_val = torch.utils.data.random_split(dset, [int(len(dset)*0.85), len(dset) - int(len(dset)*0.85)])
    val_loader = torch.utils.data.DataLoader(dset_val, batch_size = batch_size)

    model = SoTLNet(num_features=int(x.size()[1]))


    criterion = MSELoss()
    w_optimizer = SGD(model.weight_params(), lr=w_lr, momentum=0.9, weight_decay=0.1)
    # w_optimizer = torch.optim.LBFGS(model.weight_params(), lr = w_lr)
    a_optimizer = SGD(model.arch_params(), lr=a_lr, momentum=0.9, weight_decay=0.1)



    # train_bptt(num_epochs=num_epochs, model=model, dset_train=dset_train, 
    #     logging_freq=logging_freq, batch_size=batch_size, T=T, grad_clip=grad_clip)
    train_normal(num_epochs=num_epochs, model=model, dset_train=dset_train, 
        logging_freq=logging_freq, batch_size=batch_size, grad_clip=grad_clip, optim="standard")
    val_meter = valid_func(model=model, val_loader=val_loader, criterion=criterion)

