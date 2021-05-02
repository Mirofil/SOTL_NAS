# python linear/train.py --model_type=AE --dataset=isolet --arch_train_data sotl --grad_outer_loop_order=None --mode=bilevel --device=cuda --initial_degree 1 --hvp=finite_diff --epochs=100 --w_lr=0.1 --T=1 --a_lr=0.1 --hessian_tracking False --w_optim=SGD --a_optim=Adam --w_warm_start 0 --train_arch=True --a_weight_decay=0.01 --smoke_test False --dry_run=True --w_weight_decay=0.01 --batch_size=2048 --decay_scheduler None --w_scheduler None
# python linear/train.py --model_type=sigmoid --dataset=gisette --arch_train_data sotl --grad_outer_loop_order=None --mode=bilevel --device=cuda --initial_degree 1 --hvp=finite_diff --epochs=100 --w_lr=0.001 --T=1 --a_lr=0.01 --hessian_tracking False --w_optim=Adam --a_optim=Adam --w_warm_start 0 --train_arch=True --a_weight_decay=0.001 --a_decay_order 2 --smoke_test False --dry_run=True --w_weight_decay=0.001 --batch_size=64 --decay_scheduler None --loss ce
# python linear/train.py --model_type=sigmoid --dataset=gisette --arch_train_data sotl --grad_outer_loop_order=None --mode=bilevel --device=cuda --initial_degree 1 --hvp=finite_diff --epochs=100 --w_lr=0.001 --T=1 --a_lr=0.01 --hessian_tracking False --w_optim=Adam --a_optim=Adam --w_warm_start 3 --train_arch=True --a_weight_decay=0.00000001--smoke_test False --dry_run=True --w_weight_decay=0.001 --batch_size=64 --decay_scheduler None

# python linear/sotl_graph.py --model_type=max_deg --dataset=fourier --dry_run=False --T=1 --grad_outer_loop_order=1 --grad_inner_loop_order=1 --mode=bilevel --device=cpu --smoke_test True
# python linear/train.py --model_type=MNIST --dataset=MNIST --dry_run=False --T=1 --w_warm_start=0 --grad_outer_loop_order=-1 --grad_inner_loop_order=-1 --mode=bilevel --device=cuda --extra_weight_decay=0.0001 --w_weight_decay=0 --arch_train_data=val

#pip install --force git+https://github.com/Mirofil/pytorch-hessian-eigenthings.git

import itertools
import math
import os
import pickle
import time
from copy import deepcopy
from pathlib import Path
from typing import *

import fire
import numpy
import numpy as np
import scipy.linalg
import sklearn.feature_selection
import sklearn.metrics
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from hessian_eigenthings import compute_hessian_eigenthings
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from torch import Tensor
from torch.nn import Linear, MSELoss
from torch.nn import functional as F
from torch.optim import SGD, Adam
from tqdm import tqdm
from collections import defaultdict

import wandb
from datasets import get_datasets
from log_utils import AverageMeter, wandb_auth
from models import SoTLNet
from sotl_gradient import WeightBuffer, sotl_gradient
from utils import (data_generator, eval_features, featurize, hessian, jacobian,
                   prepare_seed)
from utils_features import choose_features
from utils_train import (calculate_weight_decay, compute_auc,
                         compute_train_loss, get_criterion, get_optimizers,
                         hinge_loss, reconstruction_error, switch_weights)
from utils_metrics import (ValidAccEvaluator, obtain_accuracy, SumOfWhatever)
from train_loop import valid_func, train_bptt


def main(epochs = 50,
    steps_per_epoch=None,
    batch_size = 64,
    D = 18,
    num_samples = 50000,
    w_optim='SGD',
    a_optim='SGD',
    w_decay_order=2,
    w_lr = 1e-2,
    w_momentum=0.0,
    w_weight_decay=0.0001,
    a_decay_order=2,
    a_lr = 1e-2,
    a_momentum = 0.0,
    a_weight_decay = 1,
    T = 10,
    grad_clip = 1,
    logging_freq = 200,
    w_checkpoint_freq = 1,
    max_order_y=7,
    noise_var=0.25,
    featurize_type="fourier",
    initial_degree=1,
    hvp="finite_diff",
    arch_train_data="sotl",
    normalize_a_lr=True,
    w_warm_start=0,
    extra_weight_decay=0,
    grad_inner_loop_order=-1,
    grad_outer_loop_order=-1,
    model_type="sigmoid",
    dataset="MNIST35",
    device= 'cuda' if torch.cuda.is_available() else 'cpu',
    train_arch=True,
    dry_run=False,
    hinge_loss=0.25,
    mode = "bilevel",
    hessian_tracking=False,
    smoke_test:bool = False,
    rand_seed:int = None,
    a_scheduler:str = 'step',
    w_scheduler:str = 'step',
    decay_scheduler:str=None,
    loss:str = None
    ):

    config = locals()
    if dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'
    wandb_auth()

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    try:
        __IPYTHON__
        wandb.init(project="NAS", group=f"Linear_SOTL")
    except:
        wandb.init(project="NAS", group=f"Linear_SOTL", config=config)

    if rand_seed is not None:
        prepare_seed(rand_seed)
    all_degrees = [-4, -2, 0 , 1 , 3 , 5 , 7 , 9 , 13, 17, 21] if not smoke_test else [0,2,4]
    all_metrics = {}
    for degree in tqdm(all_degrees, desc = "Iterating over degrees"):
        dataset_cfg = get_datasets(name=dataset, data_size=num_samples, max_order_generated=D,
            max_order_y=max_order_y,
            noise_var=noise_var,
            featurize_type=featurize_type)
        dset_train = dataset_cfg["dset_train"]
        dset_val = dataset_cfg["dset_val"]
        dset_test = dataset_cfg["dset_test"]
        task = dataset_cfg["task"]
        n_classes = dataset_cfg["n_classes"]
        n_features = dataset_cfg["n_features"]

        model = SoTLNet(num_features=int(len(dset_train[0][0])) if n_features is None else n_features, model_type=model_type, 
            degree=degree, weight_decay=extra_weight_decay, 
            task=task, n_classes=n_classes, config=config)
        model = model.to(device)

        criterion = get_criterion(model_type, dataset_cfg, loss)

        w_optimizer, a_optimizer, w_scheduler, a_scheduler = get_optimizers(model, config)

        metrics = train_bptt(
            epochs=epochs if not smoke_test else 4,
            steps_per_epoch=steps_per_epoch if not smoke_test else 5,
            model=model,
            criterion=criterion,
            w_optimizer=w_optimizer,
            a_optimizer=a_optimizer,
            w_scheduler=w_scheduler,
            a_scheduler=a_scheduler,
            decay_scheduler=decay_scheduler,
            dataset_cfg=dataset_cfg,
            dataset=dataset,
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
            mode=mode,
            hessian_tracking=hessian_tracking,
            )
        all_metrics[degree] = metrics

    for metric in ["train_lossEinf", "val_lossEinf"]:
        ts = [all_metrics[degree][metric][-1][-1] for degree in all_degrees]
        for i in range(len(ts)):
            wandb.log({metric:ts[i], "degree": all_degrees[i]})

        plt.plot(all_degrees, ts, label=metric)
    plt.legend()
    plt.savefig("test.jpg")
    wandb.log({"model_selection":plt})
    if model_type in ["max_deg", "softmax_mult", "linear"]:
        # lapack_solution, res, eff_rank, sing_values = scipy.linalg.lstsq(dset_train[:][0], dset_train[:][1])
        # print(f"Cond number:{abs(sing_values.max()/sing_values.min())}")

        val_meter, val_acc_meter = valid_func(model=model, dset_val=dset_val, criterion=criterion, device=device, print_results=True)

        # model.fc1.weight = torch.nn.Parameter(torch.tensor(lapack_solution).to(device))
        # model.fc1.to(device)

        # val_meter2, val_acc_meter2 = valid_func(model=model, dset_val=dset_val, criterion=criterion, device=device, print_results=False)

        # print(
        #     f"Trained val loss: {val_meter.avg}, SciPy solver val loss: {val_meter2.avg}, difference: {val_meter.avg - val_meter2.avg} (ie. {(val_meter.avg/val_meter2.avg-1)*100}% more)"
        # )
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


epochs = 75
steps_per_epoch=5
batch_size = 64
D = 18
N = 50000
w_optim='Adam'
w_decay_order=2
w_lr = 1e-3
w_momentum=0.0
w_weight_decay=0.0001
a_optim="Adam"
a_decay_order=2
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
initial_degree=15
hvp="finite_diff"
normalize_a_lr=True
w_warm_start=0
log_grad_norm=True
log_alphas=False
extra_weight_decay=0
grad_inner_loop_order=-1
grad_outer_loop_order=-1
arch_train_data="sotl"
model_type="max_deg"
dataset="fourier"
device = 'cuda'
train_arch=True
dry_run=False
mode="bilevel"
hessian_tracking=False
smoke_test=True
rand_seed = None
decay_scheduler=None
w_scheduler=None
a_scheduler=None
features=None
loss='mse'
log_suffix = ""
from copy import deepcopy
config=locals()
