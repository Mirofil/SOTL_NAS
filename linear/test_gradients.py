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


def train_model(epochs = 2,
    steps_per_epoch=None,
    batch_size = 64,
    n_features = 18,
    n_samples = 5000,
    w_optim='SGD',
    a_optim='SGD',
    w_decay_order=2,
    w_lr = 1e-2,
    w_momentum=0.0,
    w_weight_decay=0,
    a_decay_order=2,
    a_lr = 1e-2,
    a_momentum = 0.0,
    a_weight_decay = 0,
    T = 10,
    grad_clip = 100,
    logging_freq = 200,
    w_checkpoint_freq = 1,
    n_informative=7,
    noise=0.25,
    featurize_type="fourier",
    initial_degree=1,
    hvp="exact",
    ihvp="exact",
    inv_hess="exact",
    arch_train_data="sotl",
    normalize_a_lr=False,
    w_warm_start=0,
    extra_weight_decay=0,
    grad_inner_loop_order=-1,
    grad_outer_loop_order=-1,
    model_type="max_deg",
    dataset="fourier",
    device= 'cpu',
    train_arch=True,
    dry_run=True,
    hinge_loss=0.25,
    mode = "bilevel",
    hessian_tracking=False,
    smoke_test:bool = False,
    rand_seed:int = 1,
    a_scheduler:str = 'step',
    w_scheduler:str = 'step',
    decay_scheduler:str=None,
    loss:str = None,
    optimizer_mode="manual",
    bilevel_w_steps=None,
    debug=False
    ):

    config = locals()

    os.environ['WANDB_SILENT']="true"
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

    dataset_cfg = get_datasets(name=dataset, n_samples=n_samples, n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        featurize_type=featurize_type)
    dset_train = dataset_cfg["dset_train"]
    dset_val = dataset_cfg["dset_val"]
    dset_test = dataset_cfg["dset_test"]
    task = dataset_cfg["task"]
    n_classes = dataset_cfg["n_classes"]
    n_features = dataset_cfg["n_features"]

    model = SoTLNet(num_features=int(len(dset_train[0][0])) if n_features is None else n_features, model_type=model_type, 
        degree=initial_degree, weight_decay=extra_weight_decay, 
        task=task, n_classes=n_classes, config=config)
    model = model.to(device)

    criterion = get_criterion(model_type, dataset_cfg, loss)

    w_optimizer, a_optimizer, w_scheduler, a_scheduler = get_optimizers(model, config)
    model, metrics = train_bptt(
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
        ihvp=ihvp,
        inv_hess=inv_hess,
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
        optimizer_mode=optimizer_mode,
        bilevel_w_steps=bilevel_w_steps,
        debug=debug
        )
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
            true_degree = n_informative/2 
            trained_degree = model.fc1.alphas.item()
            print(f"True degree: {true_degree}, trained degree: {trained_degree}, difference: {abs(true_degree - trained_degree)}")
            print(f"Model weights: {model.fc1.weight}")
            wandb.run.summary["degree_mismatch"] = abs(true_degree-trained_degree)
        except:
            print("No model degree info; probably a different model_type was chosen")
    return model

def main():
    models = {}
    Ts = [1,5]
    arch_train_datas = ["val", "sotl"]

    for optimizer_mode in ["autograd", "manual"]:
        models[optimizer_mode] = {}
        for T in Ts:
            models[optimizer_mode][T] = {}
            for arch_train_data in arch_train_datas:
                models[optimizer_mode][T][arch_train_data] = None
                model = train_model(T=T, arch_train_data=arch_train_data, optimizer_mode=optimizer_mode, rand_seed=1)

                models[optimizer_mode][T][arch_train_data] = model.fc1.alphas[0].item()                
             
    print(models)
    


if __name__ == "__main__":
    try:
        __IPYTHON__
        main()

    except KeyboardInterrupt:
        pass
    except:
        fire.Fire(main)


