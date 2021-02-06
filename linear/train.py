import itertools
import math

import numpy
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, MSELoss
from torch.nn import functional as F
from torch.optim import SGD, Adam

import wandb
from datasets import get_datasets
from log_utils import AverageMeter, wandb_auth
from utils import (
    data_generator,
    eval_features,
    featurize,
    hessian,
    jacobian,
)
from models import SoTLNet
from sotl_utils import sotl_gradient, WeightBuffer
import scipy.linalg
import time
import fire
from utils_train import get_criterion
from tqdm import tqdm

def train_bptt(
    num_epochs: int,
    model,
    criterion,
    w_optimizer,
    a_optimizer,
    dset_train,
    dset_val,
    batch_size: int,
    T: int,
    w_checkpoint_freq: int,
    grad_clip: float,
    w_lr: float,
    logging_freq: int,
    grad_inner_loop_order: int,
    grad_outer_loop_order:int,
    hvp: str,
    arch_train_data:str,
    normalize_a_lr:bool,
    log_grad_norm:bool,
    log_alphas:bool,
    w_warm_start:int,
    extra_weight_decay:float,
    train_arch:bool,
    device:str
):
    train_loader = torch.utils.data.DataLoader(
        dset_train, batch_size=batch_size * T, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(dset_val, batch_size=batch_size)
    grad_compute_speed = AverageMeter()
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for epoch in range(num_epochs):
        model.train()

        epoch_loss = AverageMeter()
        true_batch_index = 0
        
        val_iter = iter(val_loader)
        for batch_idx, batch in enumerate(train_loader):

            
            xs, ys = torch.split(batch[0], batch_size), torch.split(
                batch[1], batch_size
            )

            weight_buffer = WeightBuffer(T=T, checkpoint_freq=w_checkpoint_freq)
            weight_buffer.add(model, 0)

            for intra_batch_idx, (x, y) in enumerate(zip(xs, ys),1):
                x = x.to(device)
                y = y.to(device)

                # weight_buffer.add(model, intra_batch_idx) # TODO Should it be added here?

                y_pred = model(x)

                param_norm = 0
                if extra_weight_decay is not None and extra_weight_decay != 0:
                    for n,weight in model.named_weight_params():
                        if 'weight' in n:
                            param_norm = param_norm + torch.pow(weight.norm(2), 2)
                    param_norm = torch.multiply(model.alpha_weight_decay, param_norm)
                # print(param_norm)
                
                
                loss = criterion(y_pred, y) + param_norm
                epoch_loss.update(loss.item())

                grads = torch.autograd.grad(
                    loss,
                    model.weight_params()
                )

                with torch.no_grad():
                    for g, w in zip(grads, model.weight_params()):
                        w.grad = g
                torch.nn.utils.clip_grad_norm_(model.weight_params(), 1)

                w_optimizer.step()
                w_optimizer.zero_grad()
                weight_buffer.add(model, intra_batch_idx)

                true_batch_index += 1
                wandb.log(
                    {
                        "Train loss": epoch_loss.avg,
                        "Epoch": epoch,
                        "Batch": true_batch_index,
                    }
                )

                if true_batch_index % logging_freq == 0:
                    print(
                        "Epoch: {}, Batch: {}, Loss: {}, Alphas: {}".format(
                            epoch,
                            true_batch_index,
                            epoch_loss.avg,
                            [x.data for x in model.arch_params()],
                        )
                    )

            if train_arch:
                val_xs = None
                val_ys = None
                if arch_train_data == "val":
                    try:
                        val_batch = next(val_iter)
                        val_xs, val_ys = torch.split(val_batch[0], batch_size), torch.split(
                            val_batch[1], batch_size
                        )

                    except:
                        val_iter = iter(val_loader)
                        val_batch = next(val_iter)
                        val_xs, val_ys = torch.split(val_batch[0], batch_size), torch.split(
                            val_batch[1], batch_size
                        )


                if epoch >= w_warm_start:
                    start_time = time.time()
                    total_arch_gradient = sotl_gradient(
                        model=model,
                        criterion=criterion,
                        xs=xs,
                        ys=ys,
                        weight_buffer=weight_buffer,
                        w_lr=w_lr,
                        hvp=hvp,
                        grad_inner_loop_order=grad_inner_loop_order,
                        grad_outer_loop_order=grad_outer_loop_order,
                        T=T,
                        normalize_a_lr=normalize_a_lr,
                        weight_decay_term=None,
                        val_xs=val_xs,
                        val_ys=val_ys
                    )
                    grad_compute_speed.update(time.time() - start_time)


                    if log_grad_norm:
                        norm = 0
                        for g in total_arch_gradient:
                            norm = norm + g.data.norm(2).item()
                        wandb.log({"Arch grad norm": norm})

                    if log_alphas:
                        if hasattr(model, "fc1") and hasattr(model.fc1, "degree"):
                            wandb.log({"Alpha":model.fc1.degree.item()})
                        if hasattr(model,"alpha_weight_decay"):
                            wandb.log({"Alpha": model.alpha_weight_decay.item()})

                    a_optimizer.zero_grad()

                    for g, w in zip(total_arch_gradient, model.arch_params()):
                        w.grad = g
                    torch.nn.utils.clip_grad_norm_(model.arch_params(), 1)
                    a_optimizer.step()

        val_results = valid_func(
            model=model, dset_val=dset_val, criterion=criterion, device=device, print_results=False
        )
        print("Epoch: {}, Val Loss: {}".format(epoch, val_results.avg))
        wandb.log({"Val loss": val_results.avg, "Epoch": epoch})
        wandb.run.summary["Grad compute speed"] = grad_compute_speed.avg

        print(f"Grad compute speed: {grad_compute_speed.avg}s")


def valid_func(model, dset_val, criterion, device = 'cuda' if torch.cuda.is_available() else 'cpu', print_results=True):
    model.eval()
    val_loader = torch.utils.data.DataLoader(dset_val, batch_size=32)
    val_meter = AverageMeter()
    val_acc_meter = AverageMeter()

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)

            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                predicted = torch.argmax(y_pred, dim=1)
                correct = torch.sum((predicted == y)).item()
                total = predicted.size()[0]
                val_acc_meter.update(correct/total)
            val_loss = criterion(y_pred, y)
            val_meter.update(val_loss.item())
    if print_results:
        print("Val loss: {}, Val acc: {}".format(val_meter.avg, val_acc_meter.avg if val_acc_meter.avg > 0 else "Not applicable"))
    return val_meter


def train_normal(
    num_epochs, model, dset_train, batch_size, grad_clip, logging_freq, optim="sgd", **kwargs
):
    train_loader = torch.utils.data.DataLoader(
        dset_train, batch_size=batch_size, shuffle=True
    )

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
                hessian_newton = torch.inverse(
                    hessian(loss * 1, linear_weight, linear_weight).reshape(
                        linear_weight.size()[1], linear_weight.size()[1]
                    )
                )
                with torch.no_grad():
                    for w in model.weight_params():
                        w = w.subtract_(torch.matmul(w.grad, hessian_newton))
            elif optim =="sgd":
                torch.nn.utils.clip_grad_norm_(model.weight_params(), 1)
                w_optimizer.step()
            else:
                raise NotImplementedError
        
            wandb.log(
                {"Train loss": epoch_loss.avg, "Epoch": epoch, "Batch": batch_idx}
            )

            if batch_idx % logging_freq == 0:
                print(
                    "Epoch: {}, Batch: {}, Loss: {}, Alphas: {}".format(
                        epoch, batch_idx, epoch_loss.avg, model.fc1.alphas.data
                    )
                )


def main(num_epochs = 50,
    batch_size = 64,
    D = 18,
    N = 50000,
    w_lr = 1e-4,
    w_momentum=0.0,
    w_weight_decay=0.0,
    a_lr = 3e-4,
    a_momentum = 0.0,
    a_weight_decay = 0.0,
    T = 10,
    grad_clip = 1,
    logging_freq = 200,
    w_checkpoint_freq = 1,
    max_order_y=7,
    noise_var=0.25,
    featurize_type="fourier",
    initial_degree=1,
    hvp="finite_diff",
    arch_train_data="val",
    normalize_a_lr=True,
    w_warm_start=0,
    extra_weight_decay=0.1e-6,
    grad_inner_loop_order=1,
    grad_outer_loop_order=-1,
    model_type="MNIST",
    dataset="MNIST",
    device= 'cuda' if torch.cuda.is_available() else 'cpu',
    train_arch=True
    ):
    config = locals()

    wandb_auth()
    wandb.init(project="NAS", group=f"Linear_SOTL", config=config)

    ### MODEL INIT
    # x, y = data_generator(N, max_order_generated=D, max_order_y=[(5,7), (9,13)], noise_var=0.25, featurize_type='fourier')
    # x, y = get_datasets("songs")

    dset_train, dset_val = get_datasets(name=dataset, data_size=N, max_order_generated=D,
        max_order_y=max_order_y,
        noise_var=noise_var,
        featurize_type=featurize_type)

    model = SoTLNet(num_features=int(len(dset_train[0][0])), model_type=model_type, degree=initial_degree, weight_decay=extra_weight_decay)
    model = model.to(device)

    criterion = get_criterion(model_type).to(device)
    w_optimizer = SGD(model.weight_params(), lr=w_lr, momentum=w_momentum, weight_decay=w_weight_decay)
    
    if train_arch:
        a_optimizer = SGD(model.arch_params(), lr=a_lr, momentum=a_momentum, weight_decay=a_weight_decay)
    else:
        # Placeholder optimizer that won't do anything - but the parameter list cannot be empty
        a_optimizer = None
    train_bptt(
        num_epochs=num_epochs,
        model=model,
        criterion=criterion,
        w_optimizer=w_optimizer,
        a_optimizer=a_optimizer,
        dset_train=dset_train,
        dset_val=dset_val,
        logging_freq=logging_freq,
        batch_size=batch_size,
        T=T,
        grad_clip=grad_clip,
        w_lr=w_lr,
        w_checkpoint_freq=w_checkpoint_freq,
        grad_inner_loop_order=grad_inner_loop_order,
        grad_outer_loop_order=grad_outer_loop_order,
        hvp=hvp,
        arch_train_data=arch_train_data,
        normalize_a_lr=normalize_a_lr,
        log_grad_norm=True,
        log_alphas=True,
        w_warm_start=w_warm_start,
        extra_weight_decay=extra_weight_decay,
        device=device,
        train_arch=train_arch
    )
    # train_normal(num_epochs=num_epochs, model=model, dset_train=dset_train,
    #     logging_freq=logging_freq, batch_size=batch_size, grad_clip=grad_clip, optim="sgd")
    if model_type in ["max_deg", "softmax_mult", "linear"]:
        lapack_solution, res, eff_rank, sing_values = scipy.linalg.lstsq(dset_train[:][0], dset_train[:][1])
        print(f"Cond number:{abs(sing_values.max()/sing_values.min())}")

        val_meter = valid_func(model=model, dset_val=dset_val, criterion=criterion, device=device, print_results=False)

        model.fc1.weight = torch.nn.Parameter(torch.tensor(lapack_solution).to(device))
        model.fc1.to(device)

        val_meter2 = valid_func(model=model, dset_val=dset_val, criterion=criterion, device=device, print_results=False)

        print(
            f"Trained val loss: {val_meter.avg}, SciPy solver val loss: {val_meter2.avg}, difference: {val_meter.avg - val_meter2.avg} (ie. {(val_meter.avg/val_meter2.avg-1)*100}% more)"
        )
        try:
            true_degree = max_order_y/2 
            trained_degree = model.fc1.alphas.item()
            print(f"True degree: {true_degree}, trained degree: {trained_degree}, difference: {abs(true_degree - trained_degree)}")
            wandb.run.summary["degree_mismatch"] = abs(true_degree-trained_degree)
        except:
            print("No model degree info; probably a different model_type was chosen")

if __name__ == "__main__":
    try:
        __IPYTHON__
        main()

    except KeyboardInterrupt:
        pass
    except:
        fire.Fire(main)


num_epochs = 50
batch_size = 64
D = 18
N = 50000
w_lr = 1e-3
w_momentum=0.0
w_weight_decay=0.0
a_lr = 3e-4
a_momentum = 0.0
a_weight_decay = 0.0
T = 10
grad_clip = 1
logging_freq = 200
w_checkpoint_freq = 1
max_order_y=7
noise_var=0.25
featurize_type="fourier"
initial_degree=1
hvp="finite_diff"
normalize_a_lr=True
w_warm_start=0
log_grad_norm=True
log_alphas=False
extra_weight_decay=0.0001
grad_inner_loop_order=1
grad_outer_loop_order=-1
arch_train_data="val"
model_type="MNIST"
dataset="MNIST"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_arch=True
config={}