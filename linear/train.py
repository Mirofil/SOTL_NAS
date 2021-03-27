# python linear/train.py --model_type=AE --dataset=isolet --arch_train_data sotl --grad_outer_loop_order=None --mode=bilevel --device=cuda --initial_degree 1 --hvp=finite_diff --epochs=100 --w_lr=0.1 --T=1 --a_lr=0.1 --hessian_tracking False --w_optim=SGD --a_optim=Adam --w_warm_start 0 --train_arch=True --a_weight_decay=0.01 --smoke_test False --dry_run=True --w_weight_decay=0.01 --batch_size=2048 --decay_scheduler None --w_scheduler None
# python linear/train.py --model_type=sigmoid --dataset=gisette --arch_train_data sotl --grad_outer_loop_order=None --mode=bilevel --device=cuda --initial_degree 1 --hvp=finite_diff --epochs=100 --w_lr=0.001 --T=1 --a_lr=0.01 --hessian_tracking False --w_optim=Adam --a_optim=Adam --w_warm_start 0 --train_arch=True --a_weight_decay=0.001 --a_decay_order 2 --smoke_test False --dry_run=True --w_weight_decay=0.001 --batch_size=64 --decay_scheduler None --loss ce
# python linear/train.py --model_type=sigmoid --dataset=gisette --arch_train_data sotl --grad_outer_loop_order=None --mode=bilevel --device=cuda --initial_degree 1 --hvp=finite_diff --epochs=100 --w_lr=0.001 --T=1 --a_lr=0.01 --hessian_tracking False --w_optim=Adam --a_optim=Adam --w_warm_start 3 --train_arch=True --a_weight_decay=0.00000001--smoke_test False --dry_run=True --w_weight_decay=0.001 --batch_size=64 --decay_scheduler None

# python linear/train.py --model_type=max_deg --dataset=fourier --dry_run=False --T=2 --grad_outer_loop_order=-1 --grad_inner_loop_order=-1 --mode=bilevel --device=cpu --optimizer_mode=manual --ihvp=exact --inv_hess=exact --hvp=exact
# python linear/train.py --model_type=max_deg --dataset=sklearn_friedman1 --dry_run=False --T=1 --a_weight_decay=0.1 --grad_outer_loop_order=1 --grad_inner_loop_order=1 --mode=bilevel --device=cpu --optimizer_mode=autograd --n_samples=50000  --epochs=1

# python linear/train.py --model_type=MNIST --dataset=MNIST --dry_run=False --T=1 --w_warm_start=0 --grad_outer_loop_order=-1 --grad_inner_loop_order=-1 --mode=bilevel --device=cuda --extra_weight_decay=0.0001 --w_weight_decay=0 --arch_train_data=val
# python linear/train.py --model_type=max_deg --epochs 20 --steps_per_epoch=1 --dataset=fourier --dry_run=True --grad_outer_loop_order=-1 --grad_inner_loop_order=-1 --mode=bilevel --device=cpu --ihvp=exact --inv_hess=exact --hvp=exact --rand_seed 1 --arch_train_data sotl --optimizer_mode=manual --T=25 --recurrent True --w_lr=1e-1 --a_lr=1e-3 --adaptive_a_lr=True
# python linear/train.py --model_type=rff_bag --epochs 50 --dataset=MNISTrff --dry_run=True --grad_outer_loop_order=-1 --grad_inner_loop_order=-1 --mode=bilevel --device=cpu --ihvp=exact --inv_hess=exact --hvp=exact --rand_seed 1 --arch_train_data sotl --optimizer_mode=autograd --loss=ce --T=2 --recurrent True --a_weight_decay 0 --a_lr=1500000000000 --w_weight_decay 0.0001 --train_arch=True --w_lr=1
# python linear/train.py --model_type=rff_bag --epochs 300 --dataset=MNISTrff --dry_run=True --grad_outer_loop_order=-1 --grad_inner_loop_order=-1 --mode=bilevel --device=cuda --ihvp=exact --inv_hess=exact --hvp=exact --rand_seed 1 --arch_train_data sotl --optimizer_mode=autograd --loss=ce --T=2 --recurrent True --a_weight_decay 0 --a_lr=1 --w_weight_decay 0.01 --train_arch=True --w_lr=10
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
    n_features = 18,
    n_samples = 5000,
    w_optim='SGD',
    a_optim='SGD',
    w_decay_order=2,
    w_lr = 1e-2,
    w_momentum=0.0,
    w_weight_decay=0.001,
    a_decay_order=2,
    a_lr = 1e-2,
    a_momentum = 0.0,
    a_weight_decay = 0.01,
    T = 10,
    grad_clip = None,
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
    loss:str = None,
    optimizer_mode="manual",
    bilevel_w_steps=None,
    debug=False,
    recurrent=True,
    l=1e5,
    adaptive_a_lr=False,
    alpha_lr = None
    ):
    if adaptive_a_lr is True:
        a_lr = a_lr*(T**(1/2))

    config = locals()
    if dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'
    if debug:
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
        task=task, n_classes=n_classes, config=config, device=device, alpha_lr=alpha_lr)
    model = model.to(device)

    criterion = get_criterion(model_type, dataset_cfg, loss)

    if alpha_lr is not None:
        config["w_lr"] = model.alpha_lr
    w_optimizer, a_optimizer, w_scheduler, a_scheduler = get_optimizers(model, config)

    model, metrics = train_bptt(
        epochs=epochs if not smoke_test else 4,
        steps_per_epoch=steps_per_epoch,
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
        debug=debug,
        recurrent=recurrent
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
    
    # else:
    #     x_train = np.array([pair[0].view(-1).numpy() for pair in dset_train])
    #     y_train = np.array([pair[1].numpy() if type(pair[1]) != int else pair[1] for pair in dset_train])

    #     x_test = np.array([pair[0].view(-1).numpy() for pair in dset_test])
    #     y_test = np.array([pair[1].numpy() if type(pair[1]) != int else pair[1] for pair in dset_test])

    #     fit_once_keys = ["MCFS", "PFA", "Lap", "PCA"]
    #     keys = ["F", "DFS-NAS", "DFS-NAS alphas", 
    #         "DFS-NAS weights", "lasso", "logistic_l1", "tree"]
    #     metrics = {"auc":{k:[] for k in [*keys, *fit_once_keys]}, "acc": {k:[] for k in [*keys, *fit_once_keys]}, "mse": {k:[] for k in [*keys, *fit_once_keys]}}
    #     AUCs = {k:[] for k in [*keys, *fit_once_keys]}
    #     accs = {k:[] for k in [*keys, *fit_once_keys]}
    #     MSEs = {k:[] for k in [*keys, *fit_once_keys]}



    #     models_to_train = {"logistic_l1":LogisticRegression(penalty='l1', solver='saga', C=1, max_iter=700 if (not smoke_test or dry_run) else 5),
    #         "tree":ExtraTreesClassifier(n_estimators = 100), 
    #         "lasso":sklearn.linear_model.Lasso()}


    #     for model_name in tqdm(models_to_train.keys(), desc="Either loading or training SKLearn models"):
    #         fname = Path(f"./checkpoints/{model_name}_{dataset}.pkl")
    #         try:
    #             with open(fname, 'rb') as f:
    #                 models_to_train[model_name] = pickle.load(f)
    #             print(f"Loaded model {models_to_train[model_name]}")
            
    #         except:
    #             print(f"Failed to load {model_name} at {str(fname)}, training instead")
    #             models_to_train[model_name].fit(x_train, y_train)
    #             try:
    #                 Path("./checkpoints").mkdir(parents=True, exist_ok=True)
    #                 if not (smoke_test or dry_run):
    #                     with open(fname, 'wb') as f:
    #                         print(f"Saving {model_name} to {str(fname)}")
    #                         pickle.dump(models_to_train[model_name], f)
    #             except:
    #                 print("Model saving failed")

    #     fit_once = {k:choose_features(model=None, x_train=x_train, x_test=x_test, y_train=y_train, top_k=100, mode = k) for k in tqdm(fit_once_keys, desc= "Fitting baseline SKFeature models")}
        
    #     models = {**models_to_train,
    #         "F":None, "DFS-NAS":model, "DFS-NAS alphas":model, "DFS-NAS weights": model, 
    #         **fit_once}

    #     to_log = {}
        
    #     if dataset_cfg['n_classes'] == 2:
    #         for k in tqdm(range(1, 100 if not smoke_test else 3), desc="Computing AUCs for different top-k features"):

    #             for key, clf_model in models.items():
    #                 if type(clf_model) is tuple:
    #                     clf_model = clf_model[0]
    #                 auc, acc = compute_auc(clf_model, k, x_train, y_train, x_test, y_test, mode = key)
    #                 metrics["auc"][key].append(auc)
    #                 metrics["acc"][key].append(auc)
    #                 AUCs[key].append(auc)
    #                 accs[key].append(acc)
    #             wandb.log({model_type:{dataset:{**{key+"_auc":AUCs[key][k-1] for key in [*keys, *fit_once_keys]},
    #                 **{key+"_acc":accs[key][k-1] for key in [*keys, *fit_once_keys]}, "k":k}}})
    #     else:
    #         for k in tqdm(range(1,100 if not smoke_test else 5, 1), desc='Computing reconstructions for MNIST-like datasets'):
    #             for key, clf_model in models.items():
    #                 if isinstance(clf_model, (tuple, list)):
    #                     clf_model = clf_model[0]
    #                 mse, acc = reconstruction_error(model=clf_model, k=k, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, mode=key)
    #                 metrics["mse"][key].append(mse)
    #                 metrics["acc"][key].append(acc)

    #                 # # We also want to examine the model perforamnce if it was retrained using only the selected features and without architecture training
    #                 # if k in [25, 50, 75] or (smoke_test and k == 1):
    #                 #     features, _, _  = choose_features(model, x_train=x_train, y_train=y_train, x_test=x_test, top_k=k, mode='normalized')
    #                 #     retrained_model = SoTLNet(num_features=k, model_type=model_type, 
    #                 #         degree=initial_degree, weight_decay=extra_weight_decay, task=task, n_classes=n_classes)
    #                 #     retrained_model.config = config
    #                 #     retrained_model = retrained_model.to(device)
    #                 #     # retrained_model.set_features(features.indices)

                        
    #                 #     criterion = get_criterion(model_type, dataset_cfg, task).to(device)

    #                 #     w_optimizer, a_optimizer, w_scheduler, a_scheduler = get_optimizers(retrained_model, config)

    #                 #     # Retrain as before BUT must set train_arch=False and change the model=retrained_model at least!
    #                 #     train_bptt(
    #                 #         epochs=20 if not smoke_test else 1,
    #                 #         steps_per_epoch=steps_per_epoch,
    #                 #         model=retrained_model,
    #                 #         criterion=criterion,
    #                 #         w_optimizer=w_optimizer,
    #                 #         a_optimizer=a_optimizer,
    #                 #         decay_scheduler=decay_scheduler,
    #                 #         w_scheduler=w_scheduler,
    #                 #         a_scheduler=a_scheduler,
    #                 #         dataset_cfg=dataset_cfg,
    #                 #         dataset=dataset,
    #                 #         dset_train=dset_train,
    #                 #         dset_val=dset_val,
    #                 #         dset_test=dset_test,
    #                 #         logging_freq=logging_freq,
    #                 #         batch_size=batch_size,
    #                 #         T=T,
    #                 #         grad_clip=grad_clip,
    #                 #         w_lr=w_lr,
    #                 #         w_checkpoint_freq=w_checkpoint_freq,
    #                 #         grad_inner_loop_order=grad_inner_loop_order,
    #                 #         grad_outer_loop_order=grad_outer_loop_order,
    #                 #         hvp=hvp,
    #                 #         arch_train_data=arch_train_data,
    #                 #         normalize_a_lr=normalize_a_lr,
    #                 #         log_grad_norm=True,
    #                 #         log_alphas=True,
    #                 #         w_warm_start=w_warm_start,
    #                 #         extra_weight_decay=extra_weight_decay,
    #                 #         device=device,
    #                 #         train_arch=False,
    #                 #         config=config,
    #                 #         mode=mode,
    #                 #         hessian_tracking=False,
    #                 #         log_suffix=f"_retrainedK={k}",
    #                 #         features=features.indices.cpu().numpy()
    #                 #     )

    #                 #     val_loss, val_acc = valid_func(retrained_model, dset_test, criterion)
    #                 #     to_log["retrained_loss"] = val_loss.avg
    #                 #     to_log["retrained_acc"] = val_acc.avg

    #             wandb.log({model_type:{dataset:{**{key+"_mse":metrics["mse"][key][k-1] for key in [*keys, *fit_once_keys]},
    #                 **{key+"_acc":metrics["acc"][key][k-1] for key in [*keys, *fit_once_keys]}, **to_log}}, "k":k})


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
n_features = 18
n_samples = 5000
w_optim='SGD'
w_decay_order=2
w_lr = 1e-3
w_momentum=0.0
w_weight_decay=0
a_optim="SGD"
a_decay_order=2
a_lr = 200
a_momentum = 0.0
a_weight_decay = 0
T = 3
grad_clip = 1
logging_freq = 200
w_checkpoint_freq = 1
n_informative=7
noise=0.25
featurize_type="fourier"
initial_degree=1
hvp="exact"
ihvp ="exact"
inv_hess="exact"
normalize_a_lr=True
w_warm_start=0
log_grad_norm=True
log_alphas=False
extra_weight_decay=0
grad_inner_loop_order=-1
grad_outer_loop_order=-1
arch_train_data="sotl"
model_type="log_reg"
dataset="fourier"
device = 'cpu'
train_arch=True
dry_run=False
mode="bilevel"
hessian_tracking=False
smoke_test=True
rand_seed = 1
decay_scheduler=None
w_scheduler=None
a_scheduler=None
features=None
loss='mse'
log_suffix = ""
optimizer_mode = "manual"
bilevel_w_steps=None
debug=False
recurrent=True
rand_seed=1
adaptive_a_lr = False
alpha_lr=0.001
arch_update_frequency=1
from copy import deepcopy
config=locals()
