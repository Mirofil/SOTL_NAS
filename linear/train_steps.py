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
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(grads, grad_clip)

        new_weights = []
        # w_optimizer.step()
        # w_optimizer.zero_grad()
        # weight_buffer.add(model, intra_batch_idx)

        # with torch.no_grad():
        for w, dw in zip(weight_buffer[-1], grads):
            new_weight = w - config["w_lr"]*dw
            # new_weight = new_weight.detach()
            new_weight.requires_grad = True
            new_weights.append(new_weight) # Manual SGD update that creates new nodes in the computational graph

        weight_buffer.direct_add(new_weights)

        model_old_weights = switch_weights(model, weight_buffer[-1]) 

    elif optimizer_mode == "autograd":
        loss, train_acc_top1 = compute_train_loss(x=x, y=y, criterion=criterion, 
            weight_buffer=weight_buffer, model=model, return_acc=True)
        grads = torch.autograd.grad(
            loss,
            weight_buffer[-1],
            create_graph=True,
        )
        # TODO should there be retain_graph = True?
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(grads, grad_clip)

        new_weights = []

        for w, dw in zip(weight_buffer[-1], grads):
            new_weights.append(w - config["w_lr"]*dw) # Manual SGD update that creates new nodes in the computational graph

        weight_buffer.direct_add(new_weights)

        # model_old_weights = switch_weights(model, weight_buffer[-1]) # This is useful for auxiliary tasks - but the actual grad evaluation happens by using the external WeightBuffer weights

    return loss, train_acc_top1

def arch_step(model, criterion, xs, ys, weight_buffer, w_lr, hvp, inv_hess, ihvp,
    grad_inner_loop_order, grad_outer_loop_order, T, 
    normalize_a_lr, val_xs, val_ys, device, grad_clip, arch_train_data,
    optimizer_mode, a_optimizer, outers, debug=False, recurrent=True):
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
            outers=outers,
            recurrent=recurrent,
            debug=debug

        )
        total_arch_gradient = arch_gradients["total_arch_gradient"]
    elif optimizer_mode == "autograd":
        arch_gradients = {}
        if grad_outer_loop_order is not None and grad_outer_loop_order > 0:
            arch_gradient_loss = sum(outers[-grad_outer_loop_order:])
        else:
            arch_gradient_loss = sum(outers)
        
        total_arch_gradient = torch.autograd.grad(arch_gradient_loss, model.arch_params(), retain_graph=True if debug else False)
        
        if debug:
            if val_xs is not None:
                x, y = val_xs[0], val_ys[0]
            else:
                x, y = xs[-1], ys[-1]
            total_arch_gradients = {}
            for idx in range(len(weight_buffer)-1):
                loss, _ = compute_train_loss(x=xs[idx], y=ys[idx], criterion=criterion, 
                    y_pred=model(xs[idx], weight=weight_buffer[idx]), model=model, return_acc=True)
                total_arch_gradient_log = torch.autograd.grad(loss, model.arch_params(), retain_graph=True if debug else False)
                total_arch_gradients[idx] = total_arch_gradient_log

            w_grad = {idx:torch.autograd.grad(weight_buffer[idx], model.arch_params(), grad_outputs=torch.ones((1,18)), retain_graph=True) for idx in range(1,len(weight_buffer))}
            test_grad= torch.zeros((1,18))
            w_grads = {}
            for idx in range(1, len(weight_buffer)):
                interim=[]
                for i in range(18):
                    test_grad[0, i] = 1
                    grads = [torch.autograd.grad(weight_buffer[idx], model.arch_params(), grad_outputs=test_grad, retain_graph=True)]
                    test_grad[0, i] = 1
                    interim.append(grads[0][0][0])
                w_grads[idx] = torch.stack(interim)

            da_direct = {}
            for idx in range(len(weight_buffer)-1):
                ws = [w.detach() for w in weight_buffer[idx]]
                for i in range(len(ws)):
                    ws[i].requires_grad=True
                loss, _ = compute_train_loss(x=xs[idx], y=ys[idx], criterion=criterion, 
                y_pred=model(xs[idx], weight=ws), model=model, return_acc=True)
                da_direct[idx]= torch.autograd.grad(loss, model.arch_params(), retain_graph=True)

            hess_matrices_dwdw = {}
            for idx in range(len(weight_buffer)-1):
                loss3 = compute_train_loss(x=xs[idx].to(device), y=ys[idx].to(device), criterion=criterion, 
                y_pred=model(xs[idx].to(device), weight_buffer[idx]), model=model)

                hess_matrices_dwdw[idx] = [hessian(loss3*1, w, w)[0] for w in weight_buffer[idx]]

            hessian_matrices_dadw = {}
            for idx in range(len(weight_buffer)-1):
                loss2 = compute_train_loss(xs[idx], ys[idx], criterion, 
                    y_pred=model(xs[idx], weight_buffer[idx]), model=model)

                hessian_matrices_dadw[idx] = [hessian(
                    loss2 * 1, weight_buffer[idx][i], arch_param
                ) for arch_param in model.arch_params() for i in range(len(weight_buffer[idx]))]
                
            dw_direct = {}
            for idx in range(len(weight_buffer)-1):
                ws = [w.detach() for w in weight_buffer[idx]]
                for i in range(len(ws)):
                    ws[i].requires_grad=True
                loss2 = compute_train_loss(xs[idx], ys[idx], criterion, 
                    y_pred=model(xs[idx], ws), model=model)
                dw_direct[idx] = torch.autograd.grad(loss2, ws)

            arch_gradients["total_arch_gradient"] = total_arch_gradient
            arch_gradients["total_arch_gradients"] = total_arch_gradients
            # arch_gradients["da_direct"] = da_direct
            # arch_gradients["dw_direct"] = dw_direct
            # arch_gradients["nested_grad"] = w_grad
            # arch_gradients["nested_grad_real"] = w_grads
            # arch_gradients["inv_hess_dwdw"] = hess_matrices_dwdw
            # arch_gradients["hess_dadw"] = hessian_matrices_dadw

        else:
            arch_gradients["total_arch_gradient"] = total_arch_gradient

    a_optimizer.zero_grad()

    for g, w in zip(total_arch_gradient, model.arch_params()):
        w.grad = g

    if grad_clip is not None:
        arch_coef = torch.nn.utils.clip_grad_norm_(model.arch_params(), grad_clip)
    a_optimizer.step()

    return arch_gradients


######## MANUAL EXAMPLE
# # Our data points
# x1 = torch.tensor([1., 2., 3., 4., 5.], requires_grad=False, dtype=torch.float32)
# x2 = torch.tensor([2., 3., 4., 5., 6.], requires_grad=False, dtype=torch.float32)
# x3 = torch.tensor([3., 4., 5., 6., 7.], requires_grad=False, dtype=torch.float32)

# model = torch.nn.Linear(5, 1)
# # We will keep copies of the model weights in a buffer as in the manual case. 
# weight1 = model.weight
# print(f"Model weight: {model.weight}")

# y1 = 2*F.linear(x1, weight1) # Loss function

# grads_y1 = torch.autograd.grad(y1, weight1, create_graph=True, retain_graph=True)

# for p, dp in zip(model.parameters(), grads_y1):
#     weight2 = weight1 - lr * dp

# print(f"Model weight after 1st update: {model.weight}")

# y2 = 2*F.linear(x2, weight2)

# grads_y2 = torch.autograd.grad(y2, weight2, create_graph=True, retain_graph=True)

# for p, dp in zip(model.parameters(), grads_y2):
#     weight3 = weight2 - lr * dp

# print(f"Model weight after 2nd update: {model.weight}")

# y3 = 2*F.linear(x3, weight3)

# grads_y3 = torch.autograd.grad(y3, lr, create_graph=True, retain_graph=True)

# grads_sotl = torch.autograd.grad(y3+y2+y1, lr, create_graph=True, retain_graph=True)
# print(grads_sotl)