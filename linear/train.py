# python linear/train.py --model_type=AE --dataset=MNISTsmall --arch_train_data sotl --grad_outer_loop_order=None --mode=bilevel --device=cuda --initial_degree 1 --hvp=finite_diff --epochs=50 --w_lr=0.0001 --T=1 --a_lr=0.001 --hessian_tracking False --w_optim=Adam --a_optim=Adam --w_warm_start 0 --train_arch=True --a_weight_decay=0.0000001 --smoke_test False --dry_run=True --w_weight_decay=1 --batch_size=64
# python linear/train.py --model_type=max_deg --dataset=fourier --dry_run=False --T=2 --grad_outer_loop_order=1 --grad_inner_loop_order=1 --mode=bilevel --device=cpu
# python linear/train.py --model_type=MNIST --dataset=MNIST --dry_run=False --T=1 --w_warm_start=0 --grad_outer_loop_order=-1 --grad_inner_loop_order=-1 --mode=bilevel --device=cuda --extra_weight_decay=0.0001 --w_weight_decay=0 --arch_train_data=val

#pip install --force git+https://github.com/Mirofil/pytorch-hessian-eigenthings.git

import os
import itertools
import math
from pathlib import Path
from copy import deepcopy

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
    prepare_seed
)
from models import SoTLNet
from sotl_gradient import sotl_gradient, WeightBuffer
import scipy.linalg
import time
import fire
from utils_train import (get_criterion, hinge_loss, get_optimizers, switch_weights, 
    compute_train_loss, calculate_weight_decay, compute_auc, 
    reconstruction_error)
from utils_features import choose_features
from tqdm import tqdm
from typing import *
from sklearn.linear_model import LogisticRegression, Lasso
import sklearn.metrics
import sklearn.feature_selection
from sklearn.ensemble import ExtraTreesClassifier
from hessian_eigenthings import compute_hessian_eigenthings
import pickle

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
    mode:str="joint",
    hessian_tracking:bool=True,
    log_suffix:str="",
    features:Sequence=None
):
    orig_model_cfg = deepcopy(model.config)
    train_loader = torch.utils.data.DataLoader(
        dset_train, batch_size=batch_size * T, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(dset_val, batch_size=batch_size)
    grad_compute_speed = AverageMeter()

    if log_alphas:
        running_degree_mismatch = 0

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for epoch in tqdm(range(epochs), desc='Iterating over epochs'):
        model.train()

        epoch_loss = AverageMeter()
        true_batch_index = 0
        
        val_iter = iter(val_loader)

        model.config["a_decay_order"] = None if (epoch < w_warm_start) else orig_model_cfg['a_decay_order']
        model.config["w_decay_order"] = None if (epoch < w_warm_start) else orig_model_cfg['w_decay_order']
        model.config['a_weight_decay'] = orig_model_cfg['a_weight_decay']*(epoch/epochs)
        model.config['w_weight_decay'] = orig_model_cfg['w_weight_decay']*(epoch/epochs)


        for batch_idx, batch in enumerate(train_loader):
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

            for intra_batch_idx, (x, y) in enumerate(zip(xs, ys),1):
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
                weight_buffer.add(model, intra_batch_idx)

                true_batch_index += 1
                to_log.update({
                        "Train loss": epoch_loss.avg,
                        "Epoch": epoch,
                        "Batch": true_batch_index,
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


            if mode == "bilevel" and epoch >= w_warm_start:

                weights_after_rollout = switch_weights(model, weight_buffer[0])
                w_optimizer, a_optimizer, w_scheduler, a_scheduler = get_optimizers(model, config)
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

        print(
            "Epoch: {}, Batch: {}, Loss: {}, Alphas: {}, Weights: {}".format(
                epoch,
                true_batch_index,
                epoch_loss.avg,
                [x.data for x in model.arch_params()] if len(str([x.data for x in model.arch_params()])) < 20 else torch.sort([x.data for x in model.arch_params()][0].view(-1), descending=True).values[0:10],
                [x.data for x in model.weight_params()] if len(str([x.data for x in model.arch_params()])) < 20 else f'Too long'
            )
        )


        val_results, val_acc_results = valid_func(
            model=model, dset_val=dset_val, criterion=criterion, device=device, print_results=False
        )

        test_results, test_acc_results = valid_func(
            model=model, dset_val=dset_test, criterion=criterion, device=device, print_results=False
        )

        auc, acc, mse, hessian_eigenvalue = None, None, None, None
        best = {"auc":{"value":0, "alphas":None}, "acc":{"value":0, "alphas":None}} # TODO this needs to be outside of the for loop to be persisent. BUt I think Ill drop it for now regardless
        if model.model_type in ['sigmoid', "MLP", "AE", "linearAE"]:
            raw_x = np.array([pair[0].view(-1).numpy() for pair in dset_train])
            raw_y = np.array([pair[1].numpy() if type(pair[1]) != int else pair[1] for pair in dset_train])

            val_x = np.array([pair[0].view(-1).numpy() for pair in dset_val])
            val_y = np.array([pair[1].numpy() if type(pair[1]) != int else pair[1] for pair in dset_val])

            if dataset in ['gisette']:
            # We need binary classification task for this to make sense
                auc, acc = compute_auc(model=model, raw_x=raw_x, raw_y=raw_y, test_x=val_x, test_y=val_y, k=25, mode="DFS-NAS alphas")
                # if auc > best["auc"]["value"]:
                #     best["auc"]["value"] = auc
                #     best["auc"]["alphas"] = model.alpha_feature_selectors
            if 'MNIST' in dataset or dataset in ['isolet', 'activity']:
                mse, acc = reconstruction_error(model=model, k=50, raw_x=raw_x, raw_y=raw_y, test_x=val_x, test_y=val_y, mode="alphas")

        if hessian_tracking:
            eigenvals, eigenvecs = compute_hessian_eigenthings(model, train_loader,
                                                    criterion, 1, full_dataset=True)
            hessian_eigenvalue = eigenvals[0]              



        print("Epoch: {}, Val Loss: {}, Test Loss: {}, Discretized AUC: {}, MSE: {}, Reconstruction Acc: {}, Hess: {}".format(epoch, val_results.avg, test_results.avg, auc, mse, acc, hessian_eigenvalue))
        to_log = {**to_log, "Val loss": val_results.avg, "Val acc": val_acc_results.avg, "Test loss": test_results.avg, "Test acc": test_acc_results.avg, "AUC_training": auc, "MSE training":mse, 
            "RecAcc training":acc, "Arch. Hessian domin. eigenvalue": hessian_eigenvalue, "Epoch": epoch}
        wandb.log({model.model_type:{dataset:{**to_log}}})
        wandb.run.summary["Grad compute speed"] = grad_compute_speed.avg

        print(f"Grad compute speed: {grad_compute_speed.avg}s")

        #NOTE this is end of epoch
        w_scheduler.step()
        if a_scheduler is not None:
            a_scheduler.step()

    # print(f"Best found metrics over validation: AUC {best['auc']['value']}")

    # if dataset in ['gisette']:
    #     # Early stopping essentially. Put back the best performing alphas for checking the top-k performances in post-train stage
    #     model.fc1.alphas = best["auc"]["alphas"]


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


def main(epochs = 5,
    steps_per_epoch=5,
    batch_size = 64,
    D = 18,
    N = 50000,
    w_optim='SGD',
    a_optim='SGD',
    w_decay_order=2,
    w_lr = 1e-2,
    w_momentum=0.0,
    w_weight_decay=0.0001,
    a_decay_order=1,
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
    initial_degree=15,
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
    hessian_tracking=True,
    auc_features_mode="normalized",
    smoke_test=False,
    rand_seed=None,
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

    if rand_seed is not None:
        prepare_seed(rand_seed)

    dataset_info = get_datasets(name=dataset, data_size=N, max_order_generated=D,
        max_order_y=max_order_y,
        noise_var=noise_var,
        featurize_type=featurize_type)
    dset_train = dataset_info["dset_train"]
    dset_val = dataset_info["dset_val"]
    dset_test = dataset_info["dset_test"]
    task = dataset_info["task"]
    n_classes = dataset_info["n_classes"]
    n_features = dataset_info["n_features"]

    model = SoTLNet(num_features=int(len(dset_train[0][0])) if n_features is None else n_features, model_type=model_type, 
        degree=initial_degree, weight_decay=extra_weight_decay, task=task, n_classes=n_classes)
    model.config = config
    model = model.to(device)

    criterion = get_criterion(model_type, task)

    w_optimizer, a_optimizer, w_scheduler, a_scheduler = get_optimizers(model, config)


    train_bptt(
        epochs=epochs if not smoke_test else 4,
        steps_per_epoch=steps_per_epoch if not smoke_test else 5,
        model=model,
        criterion=criterion,
        w_optimizer=w_optimizer,
        a_optimizer=a_optimizer,
        w_scheduler=w_scheduler,
        a_scheduler=a_scheduler,
        dataset_cfg=dataset_info,
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
        hessian_tracking=hessian_tracking
    )
    if model_type in ["max_deg", "softmax_mult", "linear"]:
        lapack_solution, res, eff_rank, sing_values = scipy.linalg.lstsq(dset_train[:][0], dset_train[:][1])
        print(f"Cond number:{abs(sing_values.max()/sing_values.min())}")

        val_meter, val_acc_meter = valid_func(model=model, dset_val=dset_val, criterion=criterion, device=device, print_results=False)

        model.fc1.weight = torch.nn.Parameter(torch.tensor(lapack_solution).to(device))
        model.fc1.to(device)

        val_meter2, val_acc_meter2 = valid_func(model=model, dset_val=dset_val, criterion=criterion, device=device, print_results=False)

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
    
    else:
        raw_x = np.array([pair[0].view(-1).numpy() for pair in dset_train])
        raw_y = np.array([pair[1].numpy() if type(pair[1]) != int else pair[1] for pair in dset_train])

        test_x = np.array([pair[0].view(-1).numpy() for pair in dset_test])
        test_y = np.array([pair[1].numpy() if type(pair[1]) != int else pair[1] for pair in dset_test])

        fit_once_keys = ["MCFS", "PFA", "Lap", "PCA"]
        keys = ["F", "DFS-NAS", "DFS-NAS alphas", "DFS-NAS weights", "lasso", "logistic_l1", "tree"]
        metrics = {"auc":{k:[] for k in [*keys, *fit_once_keys]}, "acc": {k:[] for k in [*keys, *fit_once_keys]}, "mse": {k:[] for k in [*keys, *fit_once_keys]}}
        AUCs = {k:[] for k in [*keys, *fit_once_keys]}
        accs = {k:[] for k in [*keys, *fit_once_keys]}
        MSEs = {k:[] for k in [*keys, *fit_once_keys]}



        models_to_train = {"logistic_l1":LogisticRegression(penalty='l1', solver='saga', C=1, max_iter=700 if not smoke_test else 5),
        "tree":ExtraTreesClassifier(n_estimators = 100), 
        "lasso":sklearn.linear_model.Lasso()}


        for model_name in tqdm(models_to_train.keys(), desc="Either loading or training SKLearn models"):
            fname = Path(f"./checkpoints/{model_name}_{dataset}.pkl")
            try:
                with open(fname, 'rb') as f:
                    models_to_train[model_name] = pickle.load(f)
                print(f"Loaded model {models_to_train[model_name]}")
            
            except:
                print(f"Failed to load {model_name} at {str(fname)}, training instead")
                models_to_train[model_name].fit(raw_x, raw_y)
                try:
                    Path("./checkpoints").mkdir(parents=True, exist_ok=True)
                    if not smoke_test:
                        with open(fname, 'wb') as f:
                            pickle.dump(models_to_train[model_name], f)
                except:
                    print("Model saving failed")

        fit_once = {k:choose_features(model=None, x_train=raw_x, x_test=test_x, y_train=raw_y, top_k=100, mode = k) for k in tqdm(fit_once_keys, desc= "Fitting baseline SKFeature models")}
        
        models = {**models_to_train,
            "F":None, "DFS-NAS":model, "DFS-NAS alphas":model, "DFS-NAS weights":model, 
            **fit_once}
        if dataset == 'gisette':
            for k in tqdm(range(1, 100 if not smoke_test else 3), desc="Computing AUCs for different top-k features"):

                for key, clf_model in models.items():
                    auc, acc = compute_auc(clf_model, k, raw_x, raw_y, test_x, test_y, mode = key)
                    metrics["auc"][key].append(auc)
                    metrics["acc"][key].append(auc)
                    AUCs[key].append(auc)
                    accs[key].append(acc)
                wandb.log({model_type:{dataset:{**{key+"_auc":AUCs[key][k-1] for key in [*keys, *fit_once_keys]},
                    **{key+"_acc":accs[key][k-1] for key in [*keys, *fit_once_keys]}, "k":k}}})
        
        else:

            for k in tqdm(range(1,100 if not smoke_test else 5, 1), desc='Computing reconstructions for MNIST-like datasets'):
                for key, clf_model in models.items():
                    if isinstance(clf_model, (tuple, list)):
                        clf_model = clf_model[0]
                    mse, acc = reconstruction_error(model=clf_model, k=k, raw_x=raw_x, raw_y=raw_y, test_x=test_x, test_y=test_y, mode=key)
                    metrics["mse"][key].append(mse)
                    metrics["acc"][key].append(acc)

                    # We also want to examine the model perforamnce if it was retrained using only the selected features and without architecture training
                    # if k in [9, 39, 69] or (smoke_test and k == 1):
                    #     features, _, _  = choose_features(model, top_k=k, mode='normalized')
                    #     retrained_model = SoTLNet(num_features=k, model_type=model_type, 
                    #         degree=initial_degree, weight_decay=extra_weight_decay, task=task, n_classes=n_classes)
                    #     retrained_model.config = config
                    #     retrained_model = retrained_model.to(device)
                    #     # retrained_model.set_features(features.indices)

                        
                    #     criterion = get_criterion(model_type, task).to(device)

                    #     w_optimizer, a_optimizer, w_scheduler, a_scheduler = get_optimizers(retrained_model, config)

                    #     # Retrain as before BUT must set train_arch=False and change the model=retrained_model at least!
                    #     train_bptt(
                    #         epochs=20,
                    #         steps_per_epoch=steps_per_epoch,
                    #         model=retrained_model,
                    #         criterion=criterion,
                    #         w_optimizer=w_optimizer,
                    #         a_optimizer=a_optimizer,
                    #         w_scheduler=w_scheduler,
                    #         a_scheduler=a_scheduler,
                    #         dataset=dataset,
                    #         dset_train=dset_train,
                    #         dset_val=dset_val,
                    #         dset_test=dset_test,
                    #         logging_freq=logging_freq,
                    #         batch_size=batch_size,
                    #         T=T,
                    #         grad_clip=grad_clip,
                    #         w_lr=w_lr,
                    #         w_checkpoint_freq=w_checkpoint_freq,
                    #         grad_inner_loop_order=grad_inner_loop_order,
                    #         grad_outer_loop_order=grad_outer_loop_order,
                    #         hvp=hvp,
                    #         arch_train_data=arch_train_data,
                    #         normalize_a_lr=normalize_a_lr,
                    #         log_grad_norm=True,
                    #         log_alphas=True,
                    #         w_warm_start=w_warm_start,
                    #         extra_weight_decay=extra_weight_decay,
                    #         device=device,
                    #         train_arch=False,
                    #         config=config,
                    #         mode='bilevel',
                    #         hessian_tracking=False,
                    #         log_suffix=f"_retrainedK={k}",
                    #         features=features.indices
                    #     )

                    #     val_loss, val_acc = valid_func(model, dset_test, criterion)
                    #     to_log["retrained_loss"] = val_loss.avg
                    #     to_log["retrained_acc"] = val_acc.avg

                wandb.log({model_type:{dataset:{**{key+"_mse":metrics["mse"][key][k-1] for key in [*keys, *fit_once_keys]},
                    **{key+"_acc":metrics["acc"][key][k-1] for key in [*keys, *fit_once_keys]}, "k":k}}})


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
a_optim='Adam'
a_decay_order=1
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
extra_weight_decay=0.0000
grad_inner_loop_order=-1
grad_outer_loop_order=-1
arch_train_data="sotl"
model_type="AE"
dataset="FashionMNISTsmall"
device = 'cuda'
train_arch=True
dry_run=True
mode="bilevel"
hessian_tracking=False
smoke_test=True
rand_seed = None
config=locals()
