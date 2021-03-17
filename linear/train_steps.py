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
    debug = False
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
        if True:
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
            create_graph=True
        )
        # TODO should there be retain_graph = True?
        # torch.nn.utils.clip_grad_norm_(grads, grad_clip)

        new_weights = []
        if True:
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
    optimizer_mode, a_optimizer, outers, debug=False):
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
            outers=outers

        )
        total_arch_gradient = arch_gradients["total_arch_gradient"]
    elif optimizer_mode == "autograd":
        arch_gradients = {}
        if arch_train_data == "sotl":
            arch_gradient_loss = outers[-1]
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
            total_arch_gradients = {}
            for idx in range(min(len(weight_buffer)-1, len(outers))):
                    
                total_arch_gradient_log = torch.autograd.grad(outers[idx], model.arch_params(), retain_graph=True if debug else False)
                total_arch_gradients[idx] = total_arch_gradient_log

            w_grad = {idx:torch.autograd.grad(weight_buffer[idx], model.arch_params(), grad_outputs=torch.ones((1,18)), retain_graph=True) for idx in range(1,len(weight_buffer))}
            weight_buffer[-2][0] = weight_buffer[-2][0].detach()
            weight_buffer[-2][0].requires_grad = True
            arch_gradient_loss2, _ = compute_train_loss(x=x, y=y, criterion=criterion, 
                y_pred=model(x, weight=weight_buffer[-2]), model=model, return_acc=True)
            da_direct = torch.autograd.grad(arch_gradient_loss2, model.arch_params(), retain_graph=True)
            da_direct = {}
            for idx in range(len(weight_buffer)-1):
                loss, _ = compute_train_loss(x=xs[idx], y=ys[idx], criterion=criterion, 
                y_pred=model(xs[idx], weight=weight_buffer[idx]), model=model, return_acc=True)
                da_direct[idx]= torch.autograd.grad(loss, model.arch_params(), retain_graph=True)

            # f = lambda w: compute_train_loss(x=xs[0].to(device), y=ys[0].to(device), criterion=criterion, 
            #     y_pred=model(xs[0].to(device), w), model=model)
            # k = lambda w: criterion(model(xs[2].to(device), w), ys[2].to(device))
            # mat = torch.rand((1,18), requires_grad=True)
            # mat = torch.arange(18, dtype=torch.float32).reshape(1,-1)
            # mat.requires_grad=True
            # o = lambda w: F.mse_loss(F.linear(xs[2], w), torch.rand(64,1))
            # o2 = lambda w: F.mse_loss(torch.rand(64,1), (lambda t: F.linear(xs[0], t))(w))


            # loss = o2(mat)
            # loss.backward(retain_graph=True)
            # grad_params = torch.autograd.grad(loss, mat, create_graph=True)  # p is the weight matrix for a particular layer 
            # hess_params = torch.zeros_like(grad_params[0])

            # for i in range(grad_params[0].size(0)):
            #     for j in range(grad_params[0].size(1)):
            #         hess_params[i, j] = torch.autograd.grad(grad_params[0][i][j], mat, retain_graph=True)[0][i, j]

            # hessian(f(weight_buffer[2])*1, weight_buffer[2][0])
            # l = criterion(model(xs[0].to(device), weight_buffer[0]), ys[0].to(device))


            loss3 = compute_train_loss(x=xs[0].to(device), y=ys[0].to(device), criterion=criterion, 
                y_pred=model(xs[0].to(device), weight_buffer[0]), model=model)

            hess_matrices_dwdw = [hessian(loss3*1, w, w) for w in weight_buffer[0]]

            hess_matrices_dwdw = {}
            for idx in range(len(weight_buffer)-1):
                loss3 = compute_train_loss(x=xs[idx].to(device), y=ys[idx].to(device), criterion=criterion, 
                y_pred=model(xs[idx].to(device), weight_buffer[idx]), model=model)

                hess_matrices_dwdw[idx] = [hessian(loss3*1, w, w)[0] for w in weight_buffer[idx]]


            loss2 = compute_train_loss(xs[1], ys[1], criterion, y_pred=model(x, weight_buffer[1]), model=model)

            hessian_matrices_dadw = [hessian(
                loss2 * 1, weight_buffer[1][idx], arch_param
            ) for arch_param in model.arch_params() for idx in range(len(weight_buffer[1]))]

            hessian_matrices_dadw = {}
            for idx in range(len(weight_buffer)-1):
                loss2 = compute_train_loss(xs[idx], ys[idx], criterion, 
                    y_pred=model(x, weight_buffer[idx]), model=model)

                hessian_matrices_dadw[idx] = [hessian(
                    loss2 * 1, weight_buffer[idx][i], arch_param
                ) for arch_param in model.arch_params() for i in range(len(weight_buffer[idx]))]
                

            dw_direct = torch.autograd.grad(arch_gradient_loss2, weight_buffer[-2])
            for i in range(len(weight_buffer)-1):
                loss2 = compute_train_loss(xs[idx], ys[idx], criterion, 
                    y_pred=model(x, weight_buffer[idx]), model=model)
                dw_direct = torch.autograd.grad(loss2, weight_buffer[idx])

            arch_gradients["total_arch_gradient"] = total_arch_gradient
            arch_gradients["total_arch_gradients"] = total_arch_gradients
            arch_gradients["da_direct"] = da_direct
            arch_gradients["dw_direct"] = dw_direct
            arch_gradients["nested_grad"] = w_grad
            arch_gradients["inv_hess_dwdw"] = hess_matrices_dwdw
            arch_gradients["hess_dadw"] = hessian_matrices_dadw

                
        else:
            arch_gradients["total_arch_gradient"] = total_arch_gradient

    a_optimizer.zero_grad()

    for g, w in zip(total_arch_gradient, model.arch_params()):
        w.grad = g
    
    if True:
        # arch_coef = torch.nn.utils.clip_grad_norm_(model.arch_params(), grad_clip)
        a_optimizer.step()

    return arch_gradients