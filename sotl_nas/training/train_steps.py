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
from sotl_nas.datasets.datasets import get_datasets
from sotl_nas.utils.log_utils import AverageMeter, wandb_auth
from sotl_nas.models.models import SoTLNet
from sotl_nas.sotl.gradient import WeightBuffer, sotl_gradient
from sotl_nas.utils.general_utils import (data_generator, eval_features, featurize, hessian, jacobian,
                   prepare_seed)
from sotl_nas.utils.features import choose_features
from sotl_nas.utils.train import (calculate_weight_decay, compute_auc,
                         compute_train_loss, get_criterion, get_optimizers,
                         hinge_loss, reconstruction_error, switch_weights, clip_grad_raw)
from sotl_nas.utils.metrics import (ValidAccEvaluator, obtain_accuracy, SumOfWhatever)

def train_step(x, y, criterion, model, w_optimizer, weight_buffer, grad_clip, config,
    intra_batch_idx, optimizer_mode, debug=False, detailed=False):
    # We cannot use PyTorch optimizers for AutoGrad directly because the optimizers work inplace
    if optimizer_mode == "manual":
        loss, train_acc_top1, param_norm, unreg_loss = compute_train_loss(x=x, y=y, criterion=criterion, 
            model=model, weight_buffer=weight_buffer, return_acc=True, detailed=True)
        grads = torch.autograd.grad(
            loss,
            weight_buffer[-1].values()
        )

        if grad_clip is not None:
            clip_grad_raw(grads, grad_clip)

        # new_weights = {}
        if "hyper" in model.cfg["w_optim"].lower():
            new_weights = w_optimizer.step(grads, config, weight_buffer)
            # with torch.no_grad():
            #     for (w_name, w), dw in zip(weight_buffer[-1].items(), grads):
            #         if type(config["w_lr"]) is float or not config["softplus_alpha_lr"]:
            #             new_weights[w_name] = w - config["w_lr"]*dw # Manual SGD update that creates new nodes in the computational graph
            #         else:
            #             new_weights[w_name] = w - F.softplus(config["w_lr"], config["softplus_beta"])*dw # Manual SGD update that creates new nodes in the computational graph
            #         new_weights[w_name].requires_grad = True
            weight_buffer.direct_add(new_weights)

            model_old_weights = switch_weights(model, weight_buffer[-1]) 
        else:
            with torch.no_grad():
                for g, w in zip(grads, model.weight_params()):
                    w.grad = g
            w_optimizer.step()
            w_optimizer.zero_grad()
            weight_buffer.add(model, intra_batch_idx)


    elif optimizer_mode == "autograd":
        loss, train_acc_top1, param_norm, unreg_loss = compute_train_loss(x=x, y=y, criterion=criterion, 
            weight_buffer=weight_buffer, model=model, return_acc=True, detailed=True)
        grads = torch.autograd.grad(
            loss,
            weight_buffer[-1].values(),
            create_graph=True if model.cfg["train_arch"] else False,
            retain_graph=True
        )
        # TODO should there be retain_graph = True?
        if grad_clip is not None:
            grad_coef=torch.nn.utils.clip_grad_norm_(grads, grad_clip)


        if "hyper" in model.cfg["w_optim"].lower():
            new_weights = w_optimizer.step(grads, config, weight_buffer)
            weight_buffer.direct_add(new_weights)
        else:
            with torch.no_grad():
                for g, w in zip(grads, model.weight_params()):
                    w.grad = g
            w_optimizer.step()
            # w_optimizer.zero_grad()
            weight_buffer.add(model, intra_batch_idx)
        model_old_weights = switch_weights(model, weight_buffer[-1]) # This is useful for auxiliary tasks - but the actual grad evaluation happens by using the external WeightBuffer weights

    if detailed:
        return loss, train_acc_top1, param_norm, unreg_loss
    else:
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
    if not hasattr(model, "arch_reject_count"):
        model.arch_reject_count = 0
    # if "hyper" in model.cfg["a_optim"].lower():
    if True:
        a_optimizer.zero_grad()
        for g, w in zip(total_arch_gradient, model.arch_params()):
            w.grad = g
        if grad_clip is not None:
            arch_coef = clip_grad_raw(total_arch_gradient, grad_clip)

        # if model.cfg["a_optim"].lower() == "hypersgd":
        with torch.no_grad():
            for (w_name, w), da in zip(model.named_arch_params(), total_arch_gradient):
                if "alpha_lr" in w_name and (w-model.cfg["a_lr"]*da).item() < 0:
                    if model.cfg["alpha_lr_reject_strategy"] == "half":
                        w.multiply_(1/2)
                    elif model.cfg["alpha_lr_reject_strategy"] == "zero":
                        w.multiply_(0)
                    elif model.cfg["alpha_lr_reject_strategy"] == "None" or model.cfg["alpha_lr_reject_strategy"] is None:
                        pass
                    model.arch_reject_count += 1
                else:
                    w.subtract_(other=da, alpha=model.cfg["a_lr"])

    else:
        cur_alpha_lr = None
        if hasattr(model, "alpha_lr"):
            cur_alpha_lr = model.alpha_lr.item()
        a_optimizer.zero_grad()
        for g, w in zip(total_arch_gradient, model.arch_params()):
            w.grad = g
        if grad_clip is not None:
            arch_coef = torch.nn.utils.clip_grad_norm_(model.arch_params(), grad_clip)
        a_optimizer.step()
        with torch.no_grad():
            if hasattr(model, "alpha_lr") and model.alpha_lr.item() < 0:
                if model.cfg["alpha_lr_reject_strategy"] == "half":
                    model.alpha_lr.copy_(torch.tensor(cur_alpha_lr/2))
                elif model.cfg["alpha_lr_reject_strategy"] == "zero":
                    model.alpha_lr.copy_(torch.tensor(0))
                elif model.cfg["alpha_lr_reject_strategy"] == "None" or model.cfg["alpha_lr_reject_strategy"] is None:
                    pass

                model.arch_reject_count += 1

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

# lr = torch.tensor(0.1, dtype=torch.float32, requires_grad=True) # Learning rate is the architecture parameter

# x = torch.tensor(8, requires_grad=True, dtype=torch.float32) # Weight parameter

# y1 = 2*x # Loss function

# grads_y1 = torch.autograd.grad(y1, x, create_graph=True, retain_graph=True)

# x = x - lr*grads_y1[0]

# y2 = 2*x

# grads_y2 = torch.autograd.grad(y2, x, create_graph=True, retain_graph=True)

# x = x - lr * grads_y2[0]

# y3 = 2*x

# grads_y3 = torch.autograd.grad(y3, lr, create_graph=True, retain_graph=True)

# grads_sotl = torch.autograd.grad(y3+y2+y1, lr, create_graph=True, retain_graph=True)
# print(grads_sotl)

# # Symbolically, we want: d(2x + 2(x-2a) + 2(x-2a-2a))/da 
# # In Wolfram, result is -12 (same as here)
