import itertools
import math

import numpy
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Linear, MSELoss
from torch.nn import functional as F
from torch.optim import SGD, Adam, RMSprop

import wandb
from datasets import get_datasets
from log_utils import AverageMeter, wandb_auth
from utils import (c_cov, c_tilde, data_generator, eval_features, evidence,
                   evidence_log, featurize, hessian, jacobian, sample_C,
                   sample_tau, sample_w)
from models import SoTLNet
from sotl_utils import sotl_gradient

class WeightBuffer:
    def __init__(self, checkpoint_freq, T):
        super().__init__()
        self.weight_buffer = []
        self.checkpoint_freq = checkpoint_freq
        self.T = T

    def add(self, model, intra_batch_idx):
        if intra_batch_idx % self.checkpoint_freq == 0:
            self.weight_buffer.append([w.clone() for w in model.weight_params()])
        else:
            start = math.floor(intra_batch_idx/self.checkpoint_freq)
            end = min(start+self.checkpoint_freq, self.T-1)
            self.weight_buffer.append((start, end))
    
    def __len__(self):
        return len(self.weight_buffer)
    
    def __getitem__(self, key):
        return self.get(key)

    def get(self, i:int):
        if not isinstance(self.weight_buffer[i][0], (int)):
            return self.weight_buffer[i]
        else:
            start_w = self.weight_buffer[i][0]
            end_w = self.weight_buffer[i][1]
            return [start+(end-start)/2 for (start, end) in zip(start_w, end_w)]
    
    def clear(self):
        self.weight_buffer = []

def train_bptt(num_epochs:int, model, dset_train, batch_size:int, T:int, 
    w_checkpoint_freq:int, grad_clip:float, w_lr:float, logging_freq:int, sotl_order:int, hvp:str):
    model.train()
    train_loader = torch.utils.data.DataLoader(dset_train, batch_size = batch_size*T, shuffle=True)

    for epoch in range(num_epochs):
        epoch_loss = AverageMeter()
        true_batch_index = 0
        for batch_idx, batch in enumerate(train_loader):
            xs, ys = torch.split(batch[0], batch_size), torch.split(batch[1], batch_size)

            weight_buffer = WeightBuffer(T=T, checkpoint_freq=w_checkpoint_freq)
            for intra_batch_idx, (x, y) in enumerate(zip(xs,ys)):
                weight_buffer.add(model, intra_batch_idx)

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

            total_arch_gradient = sotl_gradient(model, criterion, xs, ys, weight_buffer, w_lr=w_lr, hvp=hvp, order=sotl_order)

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
    w_checkpoint_freq=1

    ### MODEL INIT

    x, y = data_generator(N, max_order=D, noise_var=1, featurize_type='polynomial')
    # x, y = get_datasets("songs")
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    dset = torch.utils.data.TensorDataset(x, y)

    dset_train, dset_val = torch.utils.data.random_split(dset, [int(len(dset)*0.85), len(dset) - int(len(dset)*0.85)])
    val_loader = torch.utils.data.DataLoader(dset_val, batch_size = batch_size)

    model = SoTLNet(num_features=int(x.size()[1]))


    criterion = MSELoss()
    w_optimizer = SGD(model.weight_params(), lr=w_lr, momentum=0.9, weight_decay=0.1)
    a_optimizer = SGD(model.arch_params(), lr=a_lr, momentum=0.9, weight_decay=0.1)



    train_bptt(num_epochs=num_epochs, model=model, dset_train=dset_train, 
        logging_freq=logging_freq, batch_size=batch_size, T=T, 
        grad_clip=grad_clip, w_lr=w_lr, w_checkpoint_freq=w_checkpoint_freq, 
        sotl_order=1, hvp="finite_diff")
    # train_normal(num_epochs=num_epochs, model=model, dset_train=dset_train, 
    #     logging_freq=logging_freq, batch_size=batch_size, grad_clip=grad_clip, optim="standard")
    
    lapack_solution, res, eff_rank, sing_values = numpy.linalg.lstsq(x,y)
    print(f"Cond number:{abs(sing_values.max()/sing_values.min())}")
    
    val_meter = valid_func(model=model, val_loader=val_loader, criterion=criterion)

    model.fc1.weight = torch.nn.Parameter(torch.tensor(lapack_solution))

    val_meter2 = valid_func(model=model, val_loader=val_loader, criterion=criterion)

    print(f"Trained val loss: {val_meter.avg}, LAPACK solver val loss: {val_meter2.avg}, difference: {val_meter.avg - val_meter2.avg} (ie. {(val_meter.avg/val_meter2.avg-1)*100}% more)")



