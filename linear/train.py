# python linear/train.py --model_type=sigmoid --dataset=gisette --dry_run=False --grad_outer_loop_order=None --mode=bilevel --device=cpu --initial_degree 1 --hvp=finite_diff --num_epochs=75 --w_lr=0.01 --T=50
# python linear/train.py --model_type=max_deg --dataset=fourier --dry_run=False --T=2 --grad_outer_loop_order=1 --grad_inner_loop_order=1 --mode=bilevel --device=cpu
# python linear/train.py --model_type=MNIST --dataset=MNIST --dry_run=False --T=1 --w_warm_start=0 --grad_outer_loop_order=-1 --grad_inner_loop_order=-1 --mode=bilevel --device=cuda --extra_weight_decay=0.0001 --w_weight_decay=0 --arch_train_data=val

import os
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
    jacobian
)
from models import SoTLNet
from sotl_gradient import sotl_gradient, WeightBuffer
import scipy.linalg
import time
import fire
from utils_train import (get_criterion, hinge_loss, get_optimizers, switch_weights, 
compute_train_loss, calculate_weight_decay, compute_auc)
from tqdm import tqdm
from typing import *
from sklearn.linear_model import LogisticRegression, Lasso
import sklearn.metrics
import sklearn.feature_selection
from sklearn.ensemble import ExtraTreesClassifier



def train_bptt(
    num_epochs: int,
    model,
    criterion,
    w_optimizer,
    a_optimizer,
    dset_train,
    dset_val,
    dset_test,
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
    device:str,
    config: Dict,
    mode="joint",
    w_scheduler=None,
    a_scheduler=None
):
    
    train_loader = torch.utils.data.DataLoader(
        dset_train, batch_size=batch_size * T, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(dset_val, batch_size=batch_size)
    grad_compute_speed = AverageMeter()

    if log_alphas:
        running_degree_mismatch = 0

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for epoch in range(num_epochs):
        model.train()

        epoch_loss = AverageMeter()
        true_batch_index = 0
        
        val_iter = iter(val_loader)
        for batch_idx, batch in enumerate(train_loader):

            to_log = {}

            xs, ys = torch.split(batch[0], batch_size), torch.split(
                batch[1], batch_size
            )

            if mode == "bilevel":
                
                prerollout_w_optim_state_dict = w_optimizer.state_dict()

                w_scheduler2 = None

            weight_buffer = WeightBuffer(T=T, checkpoint_freq=w_checkpoint_freq)
            weight_buffer.add(model, 0)




            for intra_batch_idx, (x, y) in enumerate(zip(xs, ys),1):
                x = x.to(device)
                y = y.to(device)


                # y_pred = model(x)

                # param_norm = calculate_weight_decay(model, a_order=1, a_coef=0.0, adaptive_decay=True)

                # loss = criterion(y_pred, y) + param_norm
                loss = compute_train_loss(x=x,y=y,criterion=criterion, model=model)
                epoch_loss.update(loss.item())

                grads = torch.autograd.grad(
                    loss,
                    model.weight_params()
                )

                with torch.no_grad():
                    for g, w in zip(grads, model.weight_params()):
                        w.grad = g
                torch.nn.utils.clip_grad_norm_(model.weight_params(), grad_clip)

                w_optimizer.step()
                w_optimizer.zero_grad()
                weight_buffer.add(model, intra_batch_idx)

                true_batch_index += 1
                to_log.update({
                        "Train loss": epoch_loss.avg,
                        "Epoch": epoch,
                        "Batch": true_batch_index,
                    })

                # if true_batch_index % logging_freq == 0:
                #     print(
                #         "Epoch: {}, Batch: {}, Loss: {}, Alphas: {}".format(
                #             epoch,
                #             true_batch_index,
                #             epoch_loss.avg,
                #             [x.data for x in model.arch_params()],
                #         )
                #     )

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
                    arch_gradients = sotl_gradient(
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
                        val_ys=val_ys,
                        device=device
                    )
                    total_arch_gradient = arch_gradients["total_arch_gradient"]
                    grad_compute_speed.update(time.time() - start_time)


                    if arch_gradients['dominant_eigenvalues'] is not None:
                        # print()
                        to_log.update({"Dominant eigenvalue":arch_gradients['dominant_eigenvalues'].item()})

                    if log_grad_norm:
                        norm = 0
                        for g in total_arch_gradient:
                            norm = norm + g.data.norm(2).item()
                        to_log.update({"Arch grad norm": norm})

                    if log_alphas and batch_idx % 100 == 0:
                        if hasattr(model, "fc1") and hasattr(model.fc1, "degree"):
                            running_degree_mismatch = running_degree_mismatch + hinge_loss(model.fc1.degree.item(), config["max_order_y"]/2, config["hinge_loss"])

                            to_log.update({"Degree":model.fc1.degree.item(), "Sum of degree mismatch":running_degree_mismatch})

                        if hasattr(model,"alpha_weight_decay"):
                            to_log.update({"Alpha weight decay": model.alpha_weight_decay.item()})

                    a_optimizer.zero_grad()
  
                    for g, w in zip(total_arch_gradient, model.arch_params()):
                        w.grad = g
                    torch.nn.utils.clip_grad_norm_(model.arch_params(), grad_clip)
                    a_optimizer.step()


                    wandb.log(to_log)

            if mode == "bilevel" and epoch >= w_warm_start:

                weights_after_rollout = switch_weights(model, weight_buffer[0])
                w_optimizer, _ = get_optimizers(model, config)
                w_optimizer.load_state_dict(prerollout_w_optim_state_dict)

                #NOTE this train step should be identical to the loop above apart from WeightBuffer management! But it is difficult to abstract this in pure PyTorch, although it could be hacked with kwargs forwarding?
                num_steps = T if epoch >= w_warm_start else T
                for i in range(min(num_steps, len(xs))):
                    x,y = xs[i], ys[i]
                    x = x.to(device)
                    y = y.to(device)

                    loss = compute_train_loss(x=x,y=y,criterion=criterion, model=model)
                    epoch_loss.update(loss.item())

                    grads = torch.autograd.grad(
                        loss,
                        model.weight_params()
                    )

                    with torch.no_grad():
                        for g, w in zip(grads, model.weight_params()):
                            w.grad = g
                    torch.nn.utils.clip_grad_norm_(model.weight_params(), grad_clip)

                    w_optimizer.step()
                    w_optimizer.zero_grad()

                    to_log.update({
                            "Train loss": epoch_loss.avg,
                            "Epoch": epoch,
                            "Batch": true_batch_index,
                        })

            if true_batch_index % logging_freq == 0 or batch_idx == len(train_loader)-1:
                print(
                    "Epoch: {}, Batch: {}, Loss: {}, Alphas: {}, Weights: {}".format(
                        epoch,
                        true_batch_index,
                        epoch_loss.avg,
                        [x.data for x in model.arch_params()] if len(list(model.weight_params())) < 20 else 'Too long',
                        [x.data for x in model.weight_params()] if len(list(model.weight_params())) < 20 else 'Too long'
                    )
                )


        val_results = valid_func(
            model=model, dset_val=dset_val, criterion=criterion, device=device, print_results=False
        )

        test_results = valid_func(
            model=model, dset_val=dset_test, criterion=criterion, device=device, print_results=False
        )

        auc=None
        if model.model_type in ['sigmoid']:
            raw_x = [pair[0].numpy() for pair in dset_train]
            raw_y = [pair[1].numpy() for pair in dset_train]

            test_x = [pair[0].numpy() for pair in dset_test]
            test_y = [pair[1].numpy() for pair in dset_test]

            auc = compute_auc(model=model, raw_x=raw_x, raw_y=raw_y, test_x=test_x, test_y=test_y, k=25, mode="NAS")


        print("Epoch: {}, Val Loss: {}, Test Loss: {}, Discretized AUC: {}".format(epoch, val_results.avg, test_results.avg, auc))
        wandb.log({"Val loss": val_results.avg, "Test loss": test_results.avg, "AUC_training": auc, "Epoch": epoch})
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
            val_loss = criterion(y_pred, y.long())
            val_meter.update(val_loss.item())
    if print_results:
        print("Val loss: {}, Val acc: {}".format(val_meter.avg, val_acc_meter.avg if val_acc_meter.avg > 0 else "Not applicable"))
    return val_meter




def main(num_epochs = 5,
    batch_size = 64,
    D = 18,
    N = 50000,
    w_lr = 1e-2,
    w_momentum=0.0,
    w_weight_decay=0.001,
    a_lr = 1e-2,
    a_momentum = 0.0,
    a_weight_decay = 0.1,
    T = 10,
    grad_clip = 1,
    logging_freq = 200,
    w_checkpoint_freq = 1,
    max_order_y=7,
    noise_var=0.25,
    featurize_type="fourier",
    initial_degree=15,
    hvp="finite_diff",
    arch_train_data="sotl",
    normalize_a_lr=True,
    w_warm_start=3,
    extra_weight_decay=0,
    grad_inner_loop_order=-1,
    grad_outer_loop_order=-1,
    model_type="sigmoid",
    dataset="gisette",
    device= 'cuda' if torch.cuda.is_available() else 'cpu',
    train_arch=True,
    dry_run=False,
    hinge_loss=0.25,
    mode = "bilevel"
    ):
    config = locals()
    if dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'
    wandb_auth()

    try:
        __IPYTHON__
        wandb.init(project="NAS", group=f"Linear_SOTL")
    except:
        wandb.init(project="NAS", group=f"Linear_SOTL", config=config)


    ### MODEL INIT
    # x, y = data_generator(N, max_order_generated=D, max_order_y=[(5,7), (9,13)], noise_var=0.25, featurize_type='fourier')
    # x, y = get_datasets("songs")

    dset_train, dset_val, dset_test, task, n_classes = get_datasets(name=dataset, data_size=N, max_order_generated=D,
        max_order_y=max_order_y,
        noise_var=noise_var,
        featurize_type=featurize_type)


    model = SoTLNet(num_features=int(len(dset_train[0][0])), model_type=model_type, 
        degree=initial_degree, weight_decay=extra_weight_decay, task=task, n_classes=n_classes)
    model = model.to(device)

    criterion = get_criterion(model_type, task).to(device)

    w_optimizer, a_optimizer = get_optimizers(model, config)

    train_bptt(
        num_epochs=num_epochs,
        model=model,
        criterion=criterion,
        w_optimizer=w_optimizer,
        a_optimizer=a_optimizer,
        dset_train=dset_train,
        dset_val=dset_val,
        dset_test=dset_test,
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
        train_arch=train_arch,
        config=config,
        mode=mode
    )
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
    if model_type in ["sigmoid"]:
        keys = ["F", "NAS", "lasso", "logistic_l1", "tree"]
        AUCs = {k:[] for k in keys}
        raw_x = [pair[0].numpy() for pair in dset_train]
        raw_y = [pair[1].numpy() for pair in dset_train]

        test_x = [pair[0].numpy() for pair in dset_test]
        test_y = [pair[1].numpy() for pair in dset_test]

        clf_logistic = LogisticRegression(penalty='l1', solver='liblinear', C=1, max_iter=3000).fit(raw_x, raw_y)
        clf_tree = ExtraTreesClassifier(n_estimators = 50).fit(raw_x, raw_y)
        clf_lasso = sklearn.linear_model.Lasso().fit(raw_x,raw_y)

        for k in tqdm(range(1, 100), desc="Computing AUCs for different top-k features"):

            AUCs["NAS"].append(compute_auc(model, k, raw_x, raw_y, test_x, test_y, mode="NAS"))

            # AUCs["MI"].append(auc_score_MI)
            AUCs["F"].append(compute_auc(None, k, raw_x, raw_y, test_x, test_y, mode="F"))
            AUCs["lasso"].append(compute_auc(clf_lasso, k, raw_x, raw_y, test_x, test_y, mode ="lasso"))
            AUCs["logistic_l1"].append(compute_auc(clf_logistic, k, raw_x, raw_y, test_x, test_y, mode = "logistic_l1"))
            AUCs["tree"].append(compute_auc(clf_tree, k, raw_x, raw_y, test_x, test_y, mode = "tree"))
            wandb.log({**{key:AUCs[key][k-1] for key in keys}, "k":k})

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
w_lr = 1e-1
w_momentum=0.0
w_weight_decay=0.0
a_lr = 3e-2
a_momentum = 0.0
a_weight_decay = 0.1
T = 10
grad_clip = 1
logging_freq = 200
w_checkpoint_freq = 1
max_order_y=7
noise_var=0.25
featurize_type="fourier"
initial_degree=2
hvp="finite_diff"
normalize_a_lr=True
w_warm_start=0
log_grad_norm=True
log_alphas=False
extra_weight_decay=0.0001
grad_inner_loop_order=-1
grad_outer_loop_order=-1
arch_train_data="sotl"
model_type="sigmoid"
dataset="gisette"
device = 'cpu'
train_arch=True
dry_run=False
mode="bilevel"
config=locals()
