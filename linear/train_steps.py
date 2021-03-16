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

def train_step(x, y, criterion, model, w_optimizer, weight_buffer, grad_clip, config,
    intra_batch_idx, optimizer_mode, debug=False):

    # We cannot use PyTorch optimizers for AutoGrad directly because the optimizers work inplace
    if optimizer_mode == "manual":
        loss, train_acc_top1 = compute_train_loss(x=x, y=y, criterion=criterion, 
            model=model, weight_buffer=weight_buffer, return_acc=True)
        grads = torch.autograd.grad(
            loss,
            weight_buffer[-1]

        )
        # with torch.no_grad():
        #     for g, w in zip(grads, model.weight_params()):
        #         w.grad = g
        # torch.nn.utils.clip_grad_norm_(grads, grad_clip)

        new_weights = []
        if not debug:
            # w_optimizer.step()
            # w_optimizer.zero_grad()
            # weight_buffer.add(model, intra_batch_idx)
            with torch.no_grad():
                for w, dw in zip(weight_buffer[-1], grads):
                    new_weight = w - config["w_lr"]*dw
                    # new_weight = new_weight.detach()
                    new_weight.requires_grad = True
                    new_weights.append(new_weight) # Manual SGD update that creates new nodes in the computational graph

            weight_buffer.direct_add(new_weights)

            model_old_weights = switch_weights(model, weight_buffer[-1]) 
        else:
            weight_buffer.direct_add(weight_buffer[-1])

    elif optimizer_mode == "autograd":
        loss, train_acc_top1 = compute_train_loss(x=x, y=y, criterion=criterion, 
            weight_buffer=weight_buffer, model=model, return_acc=True)
        grads = torch.autograd.grad(
            loss,
            weight_buffer[-1],
            create_graph=True,
            retain_graph=True
        )
        # torch.nn.utils.clip_grad_norm_(grads, grad_clip)

        new_weights = []
        if not debug:
            for w, dw in zip(weight_buffer[-1], grads):
                new_weights.append(w - config["w_lr"]*dw) # Manual SGD update that creates new nodes in the computational graph

            weight_buffer.direct_add(new_weights)

            # model_old_weights = switch_weights(model, weight_buffer[-1]) # This is useful for auxiliary tasks - but the actual grad evaluation happens by using the external WeightBuffer weights
        else:
            weight_buffer.direct_add(weight_buffer[-1])
    return loss, train_acc_top1

def arch_step(model, criterion, xs, ys, weight_buffer, w_lr, hvp, inv_hess, ihvp,
    grad_inner_loop_order, grad_outer_loop_order, T, 
    normalize_a_lr, val_xs, val_ys, device, grad_clip, arch_train_data,
    optimizer_mode, a_optimizer, sotl, debug=False):
    if optimizer_mode == "manual":
        arch_gradients = sotl_gradient(
            model=model,
            criterion=criterion,
            xs=xs,
            ys=ys,
            weight_buffer=weight_buffer,
            w_lr=w_lr,
            hvp=hvp,
            inv_hess=inv_hess,
            ihvp=ihvp,
            grad_inner_loop_order=grad_inner_loop_order,
            grad_outer_loop_order=grad_outer_loop_order,
            T=T,
            normalize_a_lr=normalize_a_lr,
            weight_decay_term=None,
            val_xs=val_xs,
            val_ys=val_ys,
            device=device,

        )
        total_arch_gradient = arch_gradients["total_arch_gradient"]
    elif optimizer_mode == "autograd":
        arch_gradients = {}
        if arch_train_data == "sotl":
            arch_gradient_loss = sotl
        elif arch_train_data == "val":
            #TODO DELETE THIS DETACH ONCE DONE DEBUGGING
            # weight_buffer[-1][0] = weight_buffer[-1][0].detach()
            arch_gradient_loss, _ = compute_train_loss(x=val_xs[0], y=val_ys[0], criterion=criterion, 
                y_pred=model(val_xs[0], weight=weight_buffer[-1]), model=model, return_acc=True)
            
        total_arch_gradient = torch.autograd.grad(arch_gradient_loss, model.arch_params(), retain_graph=True if debug else False)
        
        if debug:
            if val_xs is not None:
                x, y = val_xs[0], val_ys[0]
            else:
                x, y = xs[-1], ys[-1]
            w_grad = torch.autograd.grad(weight_buffer[-2], model.arch_params(), grad_outputs=torch.ones((1,18)))
            weight_buffer[-2][0] = weight_buffer[-2][0].detach()
            weight_buffer[-2][0].requires_grad = True
            arch_gradient_loss2, _ = compute_train_loss(x=x, y=y, criterion=criterion, 
                y_pred=model(x, weight=weight_buffer[-2]), model=model, return_acc=True)
            da_direct = torch.autograd.grad(arch_gradient_loss2, model.arch_params(), retain_graph=True)
            dw_direct = torch.autograd.grad(arch_gradient_loss2, weight_buffer[-2])
            arch_gradients["da_direct"] = da_direct
            arch_gradients["dw_direct"] = dw_direct
            arch_gradients["nested_grad"] = w_grad
                

        arch_gradients["total_arch_gradient"] = total_arch_gradient

    a_optimizer.zero_grad()

    for g, w in zip(total_arch_gradient, model.arch_params()):
        w.grad = g
    
    if not debug:
        # arch_coef = torch.nn.utils.clip_grad_norm_(model.arch_params(), grad_clip)
        a_optimizer.step()

    return arch_gradients