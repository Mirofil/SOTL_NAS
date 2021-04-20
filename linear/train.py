# python linear/train.py --model_type=AE --dataset=isolet --arch_train_data sotl --grad_outer_loop_order=None --mode=bilevel --device=cuda --initial_degree 1 --hvp=finite_diff --epochs=100 --w_lr=0.1 --T=1 --a_lr=0.1 --hessian_tracking False --w_optim=SGD --a_optim=Adam --w_warm_start 0 --train_arch=True --a_weight_decay=0.01 --smoke_test False --dry_run=True --w_weight_decay=0.01 --batch_size=2048 --decay_scheduler None --w_scheduler None
# python linear/train.py --model_type=sigmoid --dataset=gisette --arch_train_data sotl --grad_outer_loop_order=None --mode=bilevel --device=cuda --initial_degree 1 --hvp=finite_diff --epochs=100 --w_lr=0.001 --T=1 --a_lr=0.01 --hessian_tracking False --w_optim=Adam --a_optim=Adam --w_warm_start 0 --train_arch=True --a_weight_decay=0.001 --a_decay_order 2 --smoke_test False --dry_run=True --w_weight_decay=0.001 --batch_size=64 --decay_scheduler None --loss ce
# python linear/train.py --model_type=sigmoid --dataset=gisette --arch_train_data sotl --grad_outer_loop_order=None --mode=bilevel --device=cuda --initial_degree 1 --hvp=finite_diff --epochs=100 --w_lr=0.001 --T=1 --a_lr=0.01 --hessian_tracking False --w_optim=Adam --a_optim=Adam --w_warm_start 3 --train_arch=True --a_weight_decay=0.00000001--smoke_test False --dry_run=True --w_weight_decay=0.001 --batch_size=64 --decay_scheduler None

# python linear/train.py --model_type=MNIST --dataset=MNIST --dry_run=False --T=1 --w_warm_start=0 --grad_outer_loop_order=-1 --grad_inner_loop_order=-1 --mode=bilevel --device=cuda --extra_weight_decay=0.0001 --w_weight_decay=0 --arch_train_data=val
# python linear/train.py --model_type=max_deg --epochs 20 --steps_per_epoch=1 --dataset=fourier --dry_run=True --grad_outer_loop_order=-1 --grad_inner_loop_order=-1 --mode=bilevel --device=cpu --ihvp=exact --inv_hess=exact --hvp=exact --rand_seed 1 --arch_train_data sotl --optimizer_mode=autograd --T=25 --recurrent True --w_lr=1e-1 --a_lr=1e-3 --adaptive_a_lr=False
# python linear/train.py --model_type=rff_bag --epochs 50 --dataset=MNISTrff --dry_run=True --grad_outer_loop_order=-1 --grad_inner_loop_order=-1 --mode=bilevel --device=cpu --ihvp=exact --inv_hess=exact --hvp=exact --rand_seed 1 --arch_train_data sotl --optimizer_mode=autograd --loss=ce --T=2 --recurrent True --a_weight_decay 0 --a_lr=1500000000000 --w_weight_decay 0.0001 --train_arch=True --w_lr=1
# python linear/train.py --model_type=rff_bag --epochs 300 --dataset=MNISTrff --dry_run=True --grad_outer_loop_order=-1 --grad_inner_loop_order=-1 --mode=bilevel --device=cuda --ihvp=exact --inv_hess=exact --hvp=exact --rand_seed 1 --arch_train_data sotl --optimizer_mode=autograd --loss=ce --T=2 --recurrent True --a_weight_decay 0 --a_lr=1 --w_weight_decay 0.01 --train_arch=True --w_lr=10

# python linear/train.py --model_type=log_reg --dataset=MNIST --dry_run=False --T=2 --w_warm_start=0 --grad_outer_loop_order=-1 --grad_inner_loop_order=-1 --mode=bilevel --device=cpu --w_weight_decay=0 --arch_train_data=sotl --alpha_lr=0.001 --w_lr=1e-3 --a_lr=1e-2 --alpha_lr=1e-3 --optimizer_mode=autograd --loss=ce --a_weight_decay=0
# python linear/train.py --cfg=linear/configs/max_deg/lin_reg.py
# python linear/train.py --cfg=linear/configs/lr/mnist_logreg.py --alpha_weight_decay=0.001 --alpha_lr=None --w_decay_order=0
# python linear/train.py --cfg=linear/configs/lr/mnist_mlp.py --alpha_lr_reject_strategy=half --T=2 --train_arch=False --w_lr=0.01 --w_optim=SGD --alpha_lr=None --mode=joint --n_samples=2000 --batch_size=16
# python linear/train.py --cfg=linear/configs/lr/mnist_mlp.py --alpha_lr_reject_strategy=half --T=15 --train_arch=True --w_lr=0.01 --w_momentum=0.9 --w_optim=HyperSGD --alpha_lr=0.01 --mode=bilevel --model_type=MLP2 --a_lr=0.01

# python linear/train.py --cfg=linear/configs/lr/mnist_vgg.py --T=3 --w_momentum=0.9 --w_optim=HyperSGD --a_lr=0.05
# python linear/train.py --cfg=linear/configs/lr/mnist_mlp.py --T=50 --a_lr=0.01 --a_optim=SGD --a_scheduler=step --grad_clip=10 --model_type="MLPLarge"

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
from train_loop import valid_func, train_bptt
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

    if config["train_arch"] is False:
        assert config["mode"] != "bilevel", "Should explicitly set mode=joint for this to make sense"

    if config["rand_seed"] is not None:
        prepare_seed(config["rand_seed"])

    if config["adaptive_a_lr"] is True:
        config["a_lr"] = config["a_lr"]*(config["T"]**(1/2))

    if config["alpha_lr"] is not None and config["softplus_alpha_lr"]:
        config["alpha_lr"] = inverse_softplus(config["alpha_lr"], config["softplus_beta"])
    dataset_cfg = get_datasets(**config)
    model = SoTLNet(cfg=config,**{**config, **dataset_cfg})
    model = model.to(config["device"])
    print(model.arch_params())

    criterion = get_criterion(config["model_type"], dataset_cfg, config["loss"])

    if config["alpha_lr"] is not None:
        assert config["train_arch"] is True
        config["w_lr"] = model.alpha_lr
    if config["alpha_weight_decay"] is not None and config["alpha_weight_decay"] != 0:
        assert config["train_arch"] is True
        config["alpha_weight_decay"] = model.alpha_weight_decay
    optim_cfg = get_optimizers(model, config)
    w_optimizer, a_optimizer, a_scheduler, w_scheduler=optim_cfg["w_optimizer"], optim_cfg["a_optimizer"], optim_cfg["a_scheduler"], optim_cfg["w_scheduler"]

    model, metrics = train_bptt(**{**dataset_cfg, **config, **optim_cfg}, model=model, criterion=criterion, dataset_cfg=dataset_cfg, config=config)
    
    if config["model_type"] in ["max_deg", "softmax_mult", "linear"]:
        # lapack_solution, res, eff_rank, sing_values = scipy.linalg.lstsq(dset_train[:][0], dset_train[:][1])
        # print(f"Cond number:{abs(sing_values.max()/sing_values.min())}")

        val_meter, val_acc_meter = valid_func(model=model, dset_val=dataset_cfg["dset_val"], criterion=criterion, device=config["device"], print_results=True)

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
