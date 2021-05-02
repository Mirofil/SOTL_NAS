# python linear/train_supernet_rff.py --cfg=linear/configs/pretrained/supernet_rff.py --T=1 --w_optim=SGD --w_lr=0.01 --grad_clip=1.0 --model_type="LinearSupernetRFF" --train_arch=False --mode=joint

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
                         hinge_loss, reconstruction_error, switch_weights, inverse_softplus)
from utils_metrics import (ValidAccEvaluator, obtain_accuracy, SumOfWhatever)
from train_loop import valid_func, train_bptt, train_supernet_rff
from configs.defaults import cfg_defaults

def main(cfg=None, **kwargs):
    config = cfg_defaults()
    if cfg is not None:
        config.merge_from_file(cfg)
    config.merge_from_list(list(itertools.chain.from_iterable(kwargs.items())))
    if config["dry_run"]:
        os.environ['WANDB_MODE'] = 'dryrun'
    if config["debug"]:
        os.environ['WANDB_SILENT']="true"
    wandb_auth()

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    try:
        __IPYTHON__
        wandb.init(project="NAS", group=f"Linear_SOTL")
    except:
        wandb.init(project="NAS", group=f"Linear_SOTL", config=config)

    if config["rand_seed"] is not None:
        prepare_seed(config["rand_seed"])

    nasbench = {"models":{}, "metrics":{}}
    for l in tqdm([10^i for i in range(20)], desc = "Iterating over lengthscales"):
        config["l"] = l

        dataset_cfg = get_datasets(**config)
        model = SoTLNet(cfg=config,**{**config, **dataset_cfg})
        model = model.to(config["device"])
        print(model.arch_params())

        criterion = get_criterion(config["model_type"], dataset_cfg, config["loss"])

        if config["alpha_lr"] is not None:
            assert config["train_arch"] is True
            config["w_lr"] = model.alpha_lr
        if config["alpha_w_momentum"] is not None:
            config["w_momentum"] = model.alpha_w_momentum
        if config["alpha_weight_decay"] is not None and config["alpha_weight_decay"] != 0:
            assert config["train_arch"] is True
            config["alpha_weight_decay"] = model.alpha_weight_decay
        optim_cfg = get_optimizers(model, config)

        model, metrics = train_supernet_rff(**{**dataset_cfg, **config, **optim_cfg}, model=model, criterion=criterion, dataset_cfg=dataset_cfg, config=config)
        nasbench[metrics][l] = metrics
        nasbench[model][l] = model

        with open('pretrained/supernet/rff.pkl', 'wb') as f:
            torch.save(nasbench, f)
            print(f"Saved LinearNASBench at iteration with lengthscale = {l}")

    

if __name__ == "__main__":
    try:
        __IPYTHON__
        main()

    except KeyboardInterrupt:
        pass
    except:
        fire.Fire(main)
