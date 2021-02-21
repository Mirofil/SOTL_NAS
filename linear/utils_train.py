import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import torch.optim
import torch.optim.lr_scheduler
import sklearn.metrics
import sklearn.feature_selection
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


def choose_features(model, top_k=20, mode='normalized'):
    if model.model_type == 'sigmoid':
        if mode == 'alphas':
            top_k = torch.topk(model.fc1.alphas, k=top_k)
        elif mode == 'normalized':
            top_k = torch.topk(model.fc1.squash_constants()*model.fc1.weight, k=top_k)

    else:
        raise NotImplementedError

    return top_k

def reconstruction_error(model, k, raw_x, raw_y, test_x, test_y, choose_features_mode = "normalized"):
    # Used to compute reconstruction errors from Concrete Autoencoder paper
    top_k = choose_features(model, top_k=k, mode=choose_features_mode)
    x = [elem[top_k.indices[0].cpu().numpy()] for elem in raw_x]
    test_x = [elem[top_k.indices[0].cpu().numpy()] for elem in test_x]
    
    clf = LinearRegression().fit(x, raw_y)
    preds = clf.predict(test_x)
    mse = ((preds-test_y)**2).mean()

    #NOTE the Concrete Autoencoder appendix says there should be 50 trees
    tree = ExtraTreesClassifier(n_estimators=50).fit(x, raw_y)
    acc = tree.score(test_x, test_y)

    return mse, acc


def compute_auc(model,k, raw_x, raw_y, test_x, test_y, mode ="F", choose_features_mode='normalized'):
    if mode in ["F", "MI", "chi2"]:
        if mode == "F":
            univ = sklearn.feature_selection.f_classif 
        elif mode == "MI":
            univ = sklearn.feature_selection.mutual_info_classif
        elif mode == "chi2":
            univ = sklearn.feature_selection.chi2

        selector = sklearn.feature_selection.SelectKBest(univ, k=k).fit(raw_x,raw_y)
        x = selector.transform(raw_x)
        test_x = selector.transform(test_x)
    
    elif mode in ["NAS", "DFS-NAS"]:
        top_k = choose_features(model, top_k=k, mode=choose_features_mode)

        x = [elem[top_k.indices[0].cpu().numpy()] for elem in raw_x]
        test_x = [elem[top_k.indices[0].cpu().numpy()] for elem in test_x]
    
    elif mode == "lasso" or mode == "logistic_l1" or mode == "tree":
        selector = sklearn.feature_selection.SelectFromModel(model, prefit=True, threshold=-np.inf, max_features=k)
        x = selector.transform(raw_x)
        test_x = selector.transform(test_x)
    
    clf = LogisticRegression(max_iter=1000).fit(x,raw_y)
    preds = clf.predict_proba(test_x)
    auc_score = sklearn.metrics.roc_auc_score(test_y, preds[:, 1])

    return auc_score   

def calculate_weight_decay(model, alpha_w_order=None, w_order=None, adaptive_decay=None, a_order=1, a_coef=1, w_coef=0.001):
    param_norm=0
    if model.alpha_weight_decay != 0 and alpha_w_order is not None:
        for n,weight in model.named_weight_params():
            if 'weight' in n:
                param_norm = param_norm + torch.pow(weight.norm(alpha_w_order), alpha_w_order)
        param_norm = torch.multiply(model.alpha_weight_decay, param_norm)

    if adaptive_decay != None and adaptive_decay != False and hasattr(model, "adaptive_weight_decay"):
        # print(model.adaptive_weight_decay())
        param_norm = param_norm + model.adaptive_weight_decay()
    
    if a_order is not None:
        if model.model_type == 'sigmoid':
            param_norm = param_norm + a_coef * torch.sum(torch.abs(torch.sigmoid(model.fc1.alphas)))
        else:
            for arch_param in model.arch_params():
                param_norm = param_norm + a_coef * torch.sum(torch.abs(arch_param))
    if w_order is not None:
        for w_param in model.weight_params():
            param_norm = param_norm + w_coef * torch.pow(torch.norm(w_param, w_order), w_order)
    
    return param_norm


def compute_train_loss(x, y, criterion, model, weight_decay=True, y_pred=None, alpha_w_order=None, w_order=1, adaptive_decay=False, a_order=1, a_coef=0.1, w_coef=0.001):
    assert model is not None or y_pred is not None

    if y_pred is None:
        y_pred = model(x)
    if weight_decay:
        param_norm = calculate_weight_decay(model, alpha_w_order=alpha_w_order, w_order=w_order,adaptive_decay=adaptive_decay, a_order=a_order, a_coef=a_coef, w_coef=w_coef)
    else:
        param_norm = 0

    loss = criterion(y_pred, y.long()) + param_norm

    return loss
def switch_weights(model, weight_buffer_elem):
    with torch.no_grad():
        old_weights = [w.clone() for w in model.weight_params()]

        for w_old, w_new in zip(model.weight_params(), weight_buffer_elem):
            w_old.copy_(w_new)
    
    return old_weights

def hinge_loss(x,y, threshold):
    if abs(x-y) <= threshold:
        return 0
    elif x > y:
        return x -y - threshold
    elif x < y:
        return y - x + threshold


def get_optimizers(model, config):
    if config["w_optim"] == 'SGD':
        w_optimizer = SGD(model.weight_params(), lr=config["w_lr"], momentum=config["w_momentum"], weight_decay=config["w_weight_decay"])
    elif config ['a_optim'] =='Adam':
        w_optimizer = Adam(model.weight_params(), lr=config["w_lr"], weight_decay=config["w_weight_decay"])

    w_scheduler = torch.optim.lr_scheduler.StepLR(w_optimizer, round(config["epochs"]/5), gamma=0.2, verbose=False)
    if config["train_arch"]:
        if config['a_optim'] == 'SGD':
            a_optimizer = SGD(model.arch_params(), lr=config["a_lr"], momentum=config["a_momentum"], weight_decay=config["a_weight_decay"])
        elif config['a_optim'] == 'Adam':
            a_optimizer = Adam(model.arch_params(), lr=config["a_lr"], weight_decay=config["a_weight_decay"])

    else:
        # Placeholder optimizer that won't do anything - but the parameter list cannot be empty
        a_optimizer = None
    
    if config["train_arch"]:
        a_scheduler = torch.optim.lr_scheduler.StepLR(a_optimizer, round(config["epochs"]/5), gamma=0.2, verbose=False)
    else:
        a_scheduler = None


    return w_optimizer, a_optimizer, w_scheduler, a_scheduler


def get_criterion(model_type, task):
    criterion=None
    if model_type in ["MNIST", "log_regression"] or task == 'clf':
        criterion = torch.nn.CrossEntropyLoss()
    elif model_type in ["max_deg", "softmax_mult", "linear", "fourier", "polynomial", "sigmoid"] or task == 'reg':
        criterion = torch.nn.MSELoss()
    
    return criterion

def train_normal(
    num_epochs, model, dset_train, batch_size, grad_clip, logging_freq, optim="sgd", **kwargs
):
    train_loader = torch.utils.data.DataLoader(
        dset_train, batch_size=batch_size, shuffle=True
    )

    model.train()
    for epoch in range(num_epochs):

        epoch_loss = AverageMeter()
        for batch_idx, batch in enumerate(train_loader):
            x, y = batch
            w_optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward(retain_graph=True)

            epoch_loss.update(loss.item())
            if optim == "newton":
                linear_weight = list(model.weight_params())[0]
                hessian_newton = torch.inverse(
                    hessian(loss * 1, linear_weight, linear_weight).reshape(
                        linear_weight.size()[1], linear_weight.size()[1]
                    )
                )
                with torch.no_grad():
                    for w in model.weight_params():
                        w = w.subtract_(torch.matmul(w.grad, hessian_newton))
            elif optim =="sgd":
                torch.nn.utils.clip_grad_norm_(model.weight_params(), 1)
                w_optimizer.step()
            else:
                raise NotImplementedError
        
            wandb.log(
                {"Train loss": epoch_loss.avg, "Epoch": epoch, "Batch": batch_idx}
            )

            if batch_idx % logging_freq == 0:
                print(
                    "Epoch: {}, Batch: {}, Loss: {}, Alphas: {}".format(
                        epoch, batch_idx, epoch_loss.avg, model.fc1.alphas.data
                    )
                )
