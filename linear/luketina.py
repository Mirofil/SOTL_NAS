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
    extra_weight_decay:float
):
    train_loader = torch.utils.data.DataLoader(
        dset_train, batch_size=batch_size * T, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(dset_val, batch_size=batch_size)
    grad_compute_speed = AverageMeter()

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
            for intra_batch_idx, (x, y) in enumerate(zip(xs, ys)):
                weight_buffer.add(model, intra_batch_idx)

                y_pred = model(x)

                param_norm = 0
                if extra_weight_decay is not None and extra_weight_decay != 0:
                    for weight in model.weight_params():
                        param_norm = param_norm + torch.pow(weight.norm(2), 2)
                
                
                loss = criterion(y_pred, y) + param_norm
                epoch_loss.update(loss.item())

                grads = torch.autograd.grad(
                    loss,
                    model.weight_params(),
                    retain_graph=True,
                    create_graph=True,
                )

                w_optimizer.zero_grad()

                with torch.no_grad():
                    for g, w in zip(grads, model.weight_params()):
                        w.grad = g
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                w_optimizer.step()
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
                    inner_loop_order=grad_inner_loop_order,
                    outer_loop_order=grad_outer_loop_order,
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

                a_optimizer.zero_grad()

                for g, w in zip(total_arch_gradient, model.arch_params()):
                    w.grad = g
                torch.nn.utils.clip_grad_norm_(model.arch_params(), 1)
                a_optimizer.step()

        val_results = valid_func(
            model=model, dset_val=dset_val, criterion=criterion, print_results=False
        )
        print("Epoch: {}, Val Loss: {}".format(epoch, val_results.avg))
        wandb.log({"Val loss": val_results.avg, "Epoch": epoch})
        wandb.run.summary["Grad compute speed"] = grad_compute_speed.avg

        print(f"Grad compute speed: {grad_compute_speed.avg}s")


def valid_func(model, dset_val, criterion, print_results=True):
    model.eval()
    val_loader = torch.utils.data.DataLoader(dset_val, batch_size=32)

    val_meter = AverageMeter()
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            y_pred = model(x)
            val_loss = criterion(y_pred, y)
            val_meter.update(val_loss.item())
    if print_results:
        print("Val loss: {}".format(val_meter.avg))
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
    w_momentum=0.9,
    w_weight_decay=0,
    a_lr = 3e-4,
    a_momentum = 0.9,
    a_weight_decay = 0,
    T = 10,
    grad_clip = 1,
    logging_freq = 200,
    w_checkpoint_freq = 1,
    max_order_y=7,
    noise_var=0.25,
    featurize_type="fourier",
    initial_degree=100,
    hvp="finite_diff",
    arch_train_data="val",
    normalize_a_lr=True,
    w_warm_start=0,
    extra_weight_decay=0.5,
    grad_inner_loop_order=-1,
    grad_outer_loop_order=-1,
    ):
    config = locals()

    wandb_auth()
    wandb.init(project="NAS", group=f"Linear_SOTL", config=config)

    ### MODEL INIT
    # x, y = data_generator(N, max_order_generated=D, max_order_y=[(5,7), (9,13)], noise_var=0.25, featurize_type='fourier')
    # x, y = get_datasets("songs")

    dset_train, dset_val = get_datasets(name="MNIST", data_size=N, max_order_generated=D,
        max_order_y=max_order_y,
        noise_var=noise_var,
        featurize_type=featurize_type)

    model = SoTLNet(num_features=int(len(dset_train[0][0])), layer_type="MNIST", degree=-1, weight_decay=extra_weight_decay)

          

    criterion = get_criterion(model_type)
    w_optimizer = SGD(model.weight_params(), lr=w_lr, momentum=w_momentum, weight_decay=w_weight_decay)
    a_optimizer = SGD(model.arch_params(), lr=a_lr, momentum=a_momentum, weight_decay=a_weight_decay)

    wandb.watch(model, log="all")
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
        extra_weight_decay=extra_weight_decay
    )
    # train_normal(num_epochs=num_epochs, model=model, dset_train=dset_train,
    #     logging_freq=logging_freq, batch_size=batch_size, grad_clip=grad_clip, optim="sgd")

    lapack_solution, res, eff_rank, sing_values = scipy.linalg.lstsq(x, y)
    print(f"Cond number:{abs(sing_values.max()/sing_values.min())}")

    val_meter = valid_func(model=model, dset_val=dset_val, criterion=criterion)

    model.fc1.weight = torch.nn.Parameter(torch.tensor(lapack_solution))

    val_meter2 = valid_func(model=model, dset_val=dset_val, criterion=criterion)

    print(
        f"Trained val loss: {val_meter.avg}, SciPy solver val loss: {val_meter2.avg}, difference: {val_meter.avg - val_meter2.avg} (ie. {(val_meter.avg/val_meter2.avg-1)*100}% more)"
    )

    true_degree = max_order_y/2 
    trained_degree = model.fc1.alphas.item()
    print(f"True degree: {true_degree}, trained degree: {trained_degree}, difference: {abs(true_degree - trained_degree)}")
    wandb.run.summary["degree_mismatch"] = abs(true_degree-trained_degree)

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
w_momentum=0.9
w_weight_decay=0.1
a_lr = 3e-3
a_momentum = 0.9
a_weight_decay = 0.2
T = 10
grad_clip = 1
logging_freq = 200
w_checkpoint_freq = 1
max_order_y=7
noise_var=0.25
featurize_type="fourier"
initial_degree=1
hvp="exact"
normalize_a_lr=True
w_warm_start=0
extra_weight_decay=1
grad_inner_loop_order=-1
grad_outer_loop_order=-1