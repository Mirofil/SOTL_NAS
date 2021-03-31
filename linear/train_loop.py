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
from train_steps import train_step, arch_step


def train_bptt(
    epochs: int,
    steps_per_epoch:int,
    model,
    criterion,
    w_optimizer,
    a_optimizer,
    w_scheduler,
    a_scheduler,
    dataset_cfg:Dict,
    dataset:str,
    dset_train,
    dset_val,
    dset_test,
    batch_size: int,
    T: int,
    w_checkpoint_freq: int,
    grad_clip: float,
    grad_clip_bilevel:float,
    w_lr: float,
    logging_freq: int,
    grad_inner_loop_order: int,
    grad_outer_loop_order:int,
    hvp: str,
    ihvp:str,
    inv_hess:str,
    arch_train_data:str,
    normalize_a_lr:bool,
    log_grad_norm:bool,
    log_alphas:bool,
    w_warm_start:int,
    extra_weight_decay:float,
    train_arch:bool,
    device:str,
    config: Dict,
    mode:str="joint",
    hessian_tracking:bool=True,
    log_suffix:str="",
    features:Sequence=None,
    decay_scheduler:str = 'linear',
    optimizer_mode="manual",
    bilevel_w_steps=None,
    debug=False,
    recurrent=True,
    arch_update_frequency=1,
    loss_threshold=None,
    **kwargs
):
    orig_model_cfg = model.cfg
    print(f"Starting with with config={model.cfg}")
    train_loader = torch.utils.data.DataLoader(
        dataset_cfg["dset_train"], batch_size=batch_size * T, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(dataset_cfg["dset_val"], batch_size=batch_size) if dataset_cfg["dset_val"] is not None else [None]
    grad_compute_speed = AverageMeter()

    suffixed_name = model.model_type + log_suffix
    running_degree_mismatch = 0

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    primal_metrics = ["train_loss", "val_loss", "train_acc", "val_acc"]
    metrics = defaultdict(lambda : [[] for _ in range(epochs)])

    val_acc_evaluator = ValidAccEvaluator(val_loader, device=device)
    arch_update_idx = 0
    
    for epoch in tqdm(range(epochs), desc='Iterating over epochs', total = epochs):
        model.train()

        train_loss = AverageMeter()
        true_batch_idx = 0
        val_iter = iter(val_loader) # Used to make sure we iterate through the whole val set with no repeats

        if decay_scheduler == 'linear':
            model.cfg["a_decay_order"] = None if (epoch < w_warm_start) else orig_model_cfg['a_decay_order']
            model.cfg["w_decay_order"] = None if (epoch < w_warm_start) else orig_model_cfg['w_decay_order']
            model.cfg['a_weight_decay'] = orig_model_cfg['a_weight_decay']*(epoch/epochs)
            model.cfg['w_weight_decay'] = orig_model_cfg['w_weight_decay']*(epoch/epochs)
        elif decay_scheduler is None or decay_scheduler == "None":
            pass

        for batch_idx, (batch, val_batch) in tqdm(enumerate(zip(train_loader, itertools.cycle(val_loader))), desc = "Iterating over batches", total = len(train_loader), disable=True):
            if steps_per_epoch is not None and batch_idx > steps_per_epoch:
                break

            to_log = {}
            xs, ys = torch.split(batch[0], batch_size), torch.split(
                batch[1], batch_size
            )
            if features is not None:
                xs = [x[:, features] for x in xs]
            if mode == "bilevel":
                
                prerollout_w_optim_state_dict = w_optimizer.state_dict()

                w_scheduler2 = None

            weight_buffer = WeightBuffer(T=T, checkpoint_freq=w_checkpoint_freq)
            weight_buffer.add(model, 0)
            losses = []
            # model.alpha_lr = torch.nn.Parameter(model.alpha_lr.detach())
            for intra_batch_idx, (x, y) in enumerate(zip(xs, ys),1):
                
                x, y = x.to(device), y.to(device)

                loss, train_acc_top1, param_norm, unreg_loss = train_step(x=x, y=y, criterion=criterion, model=model, 
                w_optimizer=w_optimizer, weight_buffer=weight_buffer, grad_clip=grad_clip_bilevel, 
                    intra_batch_idx=intra_batch_idx, config=config, optimizer_mode=optimizer_mode, debug=debug, detailed=True)
                losses.append(loss)
                if loss_threshold is not None and (loss-param_norm).item() < loss_threshold:
                    tqdm.write(f"Success at epoch {epoch}, true batch {true_batch_idx}, in total: {epoch*len(train_loader)+true_batch_idx}")

                true_batch_idx += 1
                if mode == "joint":
                    # The weight updates above were the real weight updates if using one-level optimization, so we can log them
                    metrics["train_loss"][epoch].append(-loss.item())
                    if dataset_cfg["dset_val"] is not None:
                        val_acc_top1, val_acc_top5, val_loss = val_acc_evaluator.evaluate(model, criterion)
                        metrics["val_loss"][epoch].append(-val_loss)
                        if val_acc_top1 is not None:
                            metrics["val_acc"][epoch].append(val_acc_top1)
                    else:
                        val_acc_top1, val_acc_top5, val_loss = None, None, None

                    if train_acc_top1 is not None:
                        metrics["train_acc"][epoch].append(train_acc_top1)

                    train_loss.update(loss.item())
                    to_log.update({
                            "train_loss": train_loss.avg,
                            "Epoch": epoch,
                            "Batch": true_batch_idx,
                            "arch_update_idx": arch_update_idx
                        })

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


                if epoch >= w_warm_start and batch_idx % arch_update_frequency == 0:
                    start_time = time.time()
                    arch_update_idx += 1

                    if arch_train_data == "sotl":
                        if model.cfg["sotl_agg"] is None or model.cfg["sotl_agg"] == "sum":
                            outers = losses
                        elif model.cfg["sotl_agg"] == "below_avg" or model.cfg["sotl_agg"] == "above_avg": # TODO this will only work for Autograd I think
                            outers = []
                            for loss in losses:
                                if ((loss.item() < train_loss.avg and model.cfg["sotl_agg"] == "below_avg") or (loss.item() > train_loss.avg and model.cfg["sotl_agg"] == "above_avg")) or train_loss.avg == 0:
                                    outers.append(loss)
                            if len(outers) == 0:
                                outers = losses
                        elif model.cfg["sotl_agg"] == "recent_bias":
                            outers = [losses[i] * (1.05**i) for i in range(len(losses))]
                    else:
                        outers = [compute_train_loss(x=val_x.to(device), y=val_y.to(device), criterion=criterion, 
                            y_pred=model(val_x.to(device), weight=weight_buffer[-1]), model=model) for val_x, val_y in zip(val_xs, val_ys)]

                    arch_gradients = arch_step(model=model,
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
                            val_xs=val_xs,
                            val_ys=val_ys,
                            device=device,
                            grad_clip=grad_clip,
                            a_optimizer=a_optimizer,
                            optimizer_mode=optimizer_mode,
                            arch_train_data=arch_train_data,
                            outers=outers,
                            debug=debug,
                            recurrent=recurrent)
                    total_arch_gradient = arch_gradients["total_arch_gradient"]


                    # weights_after_rollout = switch_weights(model, weight_buffer[0])

                    if debug:
                        print(f"Epoch: {epoch}, batch: {batch_idx} Arch grad: {arch_gradients}")
                    grad_compute_speed.update(time.time() - start_time)


                    if 'dominant_eigenvalues' in arch_gradients and arch_gradients['dominant_eigenvalues'] is not None:
                        # print()
                        to_log.update({"Dominant eigenvalue":arch_gradients['dominant_eigenvalues'].item()})

                    if log_grad_norm:
                        norm = 0
                        for g in total_arch_gradient:
                            norm = norm + g.data.norm(2).item()
                        to_log.update({"Arch grad norm": norm})

                    if log_alphas and batch_idx % 1 == 0:
                        if hasattr(model, "fc1") and hasattr(model.fc1, "degree"):
                            running_degree_mismatch = running_degree_mismatch + hinge_loss(model.fc1.degree.item(), config["n_informative"]/2, config["hinge_loss"])

                            to_log.update({"Degree":model.fc1.degree.item(), "Sum of degree mismatch":running_degree_mismatch})

                        if hasattr(model,"alpha_weight_decay"):
                            to_log.update({"Alpha weight decay": model.alpha_weight_decay.item()})
                        
                        if hasattr(model, "alpha_lr") and type(config["w_lr"]) is not float:

                            to_log.update({"alpha_lr": config["w_lr"].item()})
                        
                        if hasattr(model, "arch_reject_count"):
                            to_log.update({"arch_reject": model.arch_reject_count/arch_update_idx})

            if mode == "bilevel" and epoch >= w_warm_start and batch_idx % arch_update_frequency == 0:

                weights_after_rollout = switch_weights(model, weight_buffer[0])
                optims = get_optimizers(model, config)
                w_optimizer, a_optimizer, w_scheduler, a_scheduler = optims["w_optimizer"], optims["a_optimizer"], optims["w_scheduler"], optims["a_scheduler"]

                #TODO We should use new LR when doing alpha_lr setting I guess, but otherwise use the same setting I guess.. how to fix?
                # w_optimizer.load_state_dict(prerollout_w_optim_state_dict)

                #NOTE this train step should be identical to the loop above apart from WeightBuffer management! But it is difficult to abstract this in pure PyTorch, although it could be hacked with kwargs forwarding?
                if bilevel_w_steps is None or bilevel_w_steps == "None":
                    bilevel_w_steps = T

                for i in range(min(bilevel_w_steps, len(xs))):
                    x,y = xs[i], ys[i]
                    x = x.to(device)
                    y = y.to(device)

                    loss = compute_train_loss(x=x,y=y,criterion=criterion, model=model)
                    train_loss.update(loss.item())

                    grads = torch.autograd.grad(
                        loss,
                        model.weight_params()
                    )

                    with torch.no_grad():
                        for g, (w_name, w) in zip(grads, model.named_weight_params()):
                            w.grad = g
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.weight_params(), grad_clip_bilevel)

                    # with torch.no_grad():
                    #     for (w_name, w), dw in zip(model.named_weight_params(), grads):
                    #         w.subtract_(config["w_lr"]*dw)

                    w_optimizer.step()
                    w_optimizer.zero_grad()

                    metrics["train_loss"][epoch].append(-loss.item())
                    if dataset_cfg["dset_val"] is not None:
                        val_acc_top1, val_acc_top5, val_loss = val_acc_evaluator.evaluate(model, criterion)
                        metrics["val_loss"][epoch].append(-val_loss)
                        if val_acc_top1 is not None:
                            metrics["val_acc"][epoch].append(val_acc_top1)
                    else:
                        val_acc_top1, val_acc_top5, val_loss = None, None, None
                    if train_acc_top1 is not None:
                        metrics["train_acc"][epoch].append(train_acc_top1)


                    to_log.update({
                            "train_loss": train_loss.avg,
                            "Epoch": epoch,
                            "Batch": true_batch_idx,
                            "arch_update_idx": arch_update_idx
                        })

            wandb.log({suffixed_name:{dataset:{**to_log}}})
        try:
            if model.model_type == "max_deg":
                best_alphas = torch.sort([x.data for x in model.arch_params()][0].view(-1), descending=True).values[0:10]
            else:
                best_alphas = [x.data for x in model.arch_params()][0:10]

        except:
            best_alphas = "No arch params"
        tqdm.write(
            "Epoch: {}, Batch: {}, Train loss: {}, Alphas: {}, Weights: {}".format(
                epoch,
                true_batch_idx,
                train_loss.avg,
                [x.data for x in model.arch_params()] if len(str([x.data for x in model.arch_params()])) < 20 else best_alphas,
                [x.data for x in model.weight_params()] if len(str([x.data for x in model.weight_params()])) < 200 else f'Too long'
            )
        )
        # if debug:
        #     tqdm.write(
        #         "Epoch: {}, Arch direct_da: {}, Arch direct_dw: {}".format(
        #             epoch,
        #             arch_gradients["da_direct"],
        #             arch_gradients["dw_direct"]
        #         )
        #     )

        # Check performance of model on val/test sets for logging only
        val_results, val_acc_results = valid_func(
            model=model, dset_val=dataset_cfg["dset_val"], criterion=criterion, device=device, 
            print_results=False, features=features
        ) if dataset_cfg["dset_val"] is not None else (AverageMeter(), AverageMeter())

        test_results, test_acc_results = valid_func(
            model=model, dset_val=dataset_cfg["dset_test"], criterion=criterion, device=device, 
            print_results=False, features=features
        )

        # Doesnt do anything if not doing feature selection
        auc, acc, mse, hessian_eigenvalue = eval_feature_selection(model, dataset_cfg["dset_train"], dataset_cfg["dset_val"], dataset_cfg)

        if hessian_tracking:
            eigenvals, eigenvecs = compute_hessian_eigenthings(model, train_loader,
                                                    criterion, num_eigenthings=1, full_dataset=True)
            hessian_eigenvalue = eigenvals[0]              

        val_results_loss, val_results_acc = val_results.avg if val_results is not None else None, val_acc_results.avg if val_acc_results is not None else None
        tqdm.write("Epoch: {}, Val Loss: {}, Test Loss: {}, Discretized AUC: {}, MSE: {}, Reconstruction Acc: {}, Hess: {}".format(epoch, val_results.avg, test_results.avg, auc, mse, acc, hessian_eigenvalue))
        to_log = {**to_log, "val_loss": val_results_loss, "val_acc": val_results_acc, "test_loss": test_results.avg, "test_acc": test_acc_results.avg, "AUC_training": auc, "MSE training":mse, 
            "RecAcc training":acc, "Arch. Hessian domin. eigenvalue": hessian_eigenvalue, "epoch": epoch, "arch_update_idx": arch_update_idx}
        to_log = {k:v for k,v in to_log.items() if v is not None}
        wandb.log({suffixed_name:{dataset:{**to_log}}})
        wandb.run.summary["Grad compute speed"] = grad_compute_speed.avg

        tqdm.write(f"Grad compute speed: {grad_compute_speed.avg}s")

        #NOTE this is end of epoch
        if w_scheduler is not None:
            w_scheduler.step()
        if a_scheduler is not None:
            a_scheduler.step()

    try:
        for metric in primal_metrics:
            if metric in metrics.keys():
                metrics[metric+"E1"] = SumOfWhatever(measurements = metrics[metric], e=1).get_time_series(chunked=True)
                metrics[metric+"Einf"] = SumOfWhatever(measurements = metrics[metric], e=1000).get_time_series(chunked=True)
        
        for epoch in range(epochs):
            for batch in range(len(train_loader)*T):
                all_data = {}
                for metric in metrics.keys():
                    if not ("E1" in metric or "Einf" in metric) or batch >= len(metrics[metric][epoch]):
                        continue
                    all_data[metric] = metrics[metric][epoch][batch]
                wandb.log({"epoch":epoch, "true_step":batch, **all_data})
    except Exception as e:
        print(e)
        print(f"Final logging of SoTL-ish metrics failed")

    return model, metrics

def valid_func(model, dset_val, criterion, 
    device = 'cuda' if torch.cuda.is_available() else 'cpu', print_results=True,
    features=None):
    model.eval()
    val_loader = torch.utils.data.DataLoader(dset_val, batch_size=32)
    val_meter = AverageMeter()
    val_acc_meter = AverageMeter()

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            if features is not None:
                x = x[:, features]
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)

            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                predicted = torch.argmax(y_pred, dim=1)
                correct = torch.sum((predicted == y)).item()
                total = predicted.size()[0]
                val_acc_meter.update(correct/total)
            if type(criterion) is torch.nn.modules.loss.MSELoss:
                # TODO The reshape seems to be necessary for auto-encoder - not sure why the shape is still mangled up even though I tried to reshape when returning from AE
                if "AE" in model.model_type:
                    val_loss = criterion(y_pred, x)
                else:
                    val_loss = criterion(y_pred, y)

            elif type(criterion) is torch.nn.CrossEntropyLoss:
                val_loss = criterion(y_pred, y.long()) 

            val_meter.update(val_loss.item())
    if print_results:
        print("Val loss: {}, Val acc: {}".format(val_meter.avg, val_acc_meter.avg if val_acc_meter.avg > 0 else "Not applicable"))

    model.train()
    return val_meter, val_acc_meter


def eval_feature_selection(model, dset_train, dset_val, dataset_cfg):

    auc, acc, mse, hessian_eigenvalue = None, None, None, None
    best = {"auc":{"value":0, "alphas":None}, "acc":{"value":0, "alphas":None}} # TODO this needs to be outside of the for loop to be persisent. BUt I think Ill drop it for now regardless
    if model.model_type in ['sigmoid', "MLP", "AE", "linearAE", "pt_logistic_l1", "log_regression"]:
        # This is for the feature selection task
        x_train = np.array([pair[0].view(-1).numpy() for pair in dset_train])
        y_train = np.array([pair[1].numpy() if type(pair[1]) != int else pair[1] for pair in dset_train])

        x_val = np.array([pair[0].view(-1).numpy() for pair in dset_val])
        y_val = np.array([pair[1].numpy() if type(pair[1]) != int else pair[1] for pair in dset_val])

        if dataset_cfg['n_classes'] == 2:
        # We need binary classification task for this to make sense
            auc, acc = compute_auc(model=model, raw_x=x_train, raw_y=y_train, test_x=x_val, test_y=y_val, k=25, mode="DFS-NAS alphas")
            # if auc > best["auc"]["value"]:
            #     best["auc"]["value"] = auc
            #     best["auc"]["alphas"] = model.alpha_feature_selectors
        else:
            mse, acc = reconstruction_error(model=model, k=50, x_train=x_train, y_train=y_train, x_test=x_val, y_test=y_val, mode="alphas")
    return auc, acc, mse, hessian_eigenvalue
