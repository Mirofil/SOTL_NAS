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
from traits import FeatureSelectableTrait, AutoEncoder
import torch.nn.functional as F
from sklearn.decomposition import PCA
from utils_features import mcfs_ours, pfa_transform, lap_ours, pca, univariate_test, sklearn_model, choose_features
from utils_metrics import obtain_accuracy
from sotl_optimizers import HyperSGD
import operator
from torch._six import inf


def reconstruction_error(model, k, x_train, y_train, x_test, 
    y_test, mode = "normalized"):
    # Used to compute reconstruction errors from Concrete Autoencoder paper
    indices, x, x_test = choose_features(model=model, x_train=x_train, 
        x_test=x_test, y_train=y_train, top_k=k, mode=mode)
    
    clf = LinearRegression().fit(x, y_train)
    preds = clf.predict(x_test)
    mse = ((preds-y_test)**2).mean()

    #NOTE the Concrete Autoencoder appendix says there should be 50 trees
    tree = ExtraTreesClassifier(n_estimators=50).fit(x, y_train)
    acc = tree.score(x_test, y_test)

    return mse, acc


def compute_auc(model, k, raw_x, raw_y, test_x, test_y, mode ="F", verbose=True):
    indices, x, test_x = choose_features(model=model, x_train=raw_x, y_train=raw_y, 
        x_test=test_x, mode=mode, top_k=k)

    clf = LogisticRegression(max_iter=1000).fit(x,raw_y)
    preds = clf.predict_proba(test_x)
    auc_score = sklearn.metrics.roc_auc_score(test_y, preds[:, 1])
    acc = clf.score(test_x, test_y)

    return auc_score, acc

def calculate_weight_decay(model, alpha_w_order=None, w_order=1, adaptive_decay=None, 
    a_order=1, a_coef=1, w_coef=0.001):
    param_norm=0
    param_norm_a = 0
    param_norm_w = 0
    D_a = 0
    D_w = 0
    # TODO think about how to do param_norms in this alpha_weight_decay and adaptive_decay setting
    if model.alpha_weight_decay != 0 and alpha_w_order is not None:
        for n,weight in model.named_weight_params():
            if 'weight' in n:
                param_norm = param_norm + torch.pow(weight.norm(alpha_w_order), alpha_w_order)
        param_norm = torch.multiply(model.alpha_weight_decay, param_norm)

    if adaptive_decay != None and adaptive_decay != False and hasattr(model, "adaptive_weight_decay"):
        param_norm = param_norm + model.adaptive_weight_decay()
    
    if a_order is not None and model.cfg["train_arch"]:
        if model.model_type in ['sigmoid']:
            for arch_param in model.arch_params():
                param_norm_a = param_norm_a + a_coef * torch.sum(torch.norm(torch.sigmoid(arch_param), a_order))
                D_a = D_a + torch.numel(arch_param)
        
        elif model.model_type in ['max_deg']:
            for arch_param in model.arch_params():
                param_norm_a = param_norm_a + a_coef * torch.sum(torch.norm(arch_param, a_order))
                D_a = D_a + torch.numel(arch_param)
        else:
            if hasattr(model.model, "squash"):
                for arch_param in model.arch_params():
                    param_norm_a = param_norm_a + a_coef * torch.sum(torch.norm(model.model.squash(arch_param), a_order))
                    D_a = D_a + torch.numel(arch_param)
            else:
                for arch_param in model.arch_params():
                    param_norm_a = param_norm_a + a_coef * torch.sum(torch.norm(arch_param, a_order))
                    D_a = D_a + torch.numel(arch_param)
    if w_order is not None:
        for w_param in model.weight_params():
            param_norm_w = param_norm_w + w_coef * torch.pow(torch.norm(w_param, w_order), w_order)
            D_w = D_w + torch.numel(w_param)
    param_norm = param_norm_a/max(D_a, 1) + param_norm_w/max(D_w, 1)
    # param_norm = param_norm_a + param_norm_w

    return param_norm


def compute_train_loss(x, y, criterion, model, weight_buffer=None, weight_decay=True, 
    y_pred=None, alpha_w_order=None, w_order=None, adaptive_decay=False, 
    a_order=None, a_coef=None, w_coef=None, return_acc=False, debug=False, detailed=False):
    assert model is not None or y_pred is not None
    assert y_pred is None or weight_buffer is None

    if y_pred is None:
        if weight_buffer is None:
            y_pred = model(x)
        else:
            y_pred = model(x, weight=weight_buffer[-1])
        
    if issubclass(type(model.model), AutoEncoder) or 'AE' in model.model_type:
        y = x
        assert y_pred.shape == x.shape

    if weight_decay:
        param_norm = calculate_weight_decay(model, alpha_w_order=alpha_w_order, w_order=model.cfg["w_decay_order"],
            adaptive_decay=adaptive_decay, a_order=model.cfg["a_decay_order"], 
            a_coef=model.cfg["a_weight_decay"], w_coef=model.cfg["w_weight_decay"])
    else:
        param_norm = 0

    if type(criterion) is torch.nn.modules.loss.MSELoss:
        unreg_loss = criterion(y_pred, y)

    elif type(criterion) is torch.nn.CrossEntropyLoss:
        unreg_loss = criterion(y_pred, y.long())
    loss = unreg_loss + param_norm

    if debug:
        print(f"Train loss: {loss}, param_norm: {param_norm}")

    if return_acc:
        if y_pred.shape[1] != 1: # Must be regression task
            acc_top1 = obtain_accuracy(y_pred.cpu().data, y.cpu().data, topk=(1,))
        else:
            acc_top1 = None
        if detailed:
            return loss, acc_top1[0].item() if acc_top1 is not None else None, param_norm, unreg_loss
        else:
            return loss, acc_top1[0].item() if acc_top1 is not None else None
    else:
        if detailed:
            return loss, param_norm, unreg_loss
        return loss
def clip_grad_raw(parameters, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].device
    if norm_type == inf:
        total_norm = max(p.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.detach(), norm_type).to(device) for p in parameters]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.detach().mul_(clip_coef.to(p.device))
    return total_norm

def record_parents(model, root=""):
    children = dict(model.named_children())
    for module_name, module in children.items():
        first_part = (root + ".") if root != "" else ""
        cur_path = first_part + module_name
        module.parent_path = cur_path
        record_parents(module, root = cur_path)
    pass

def switch_weights(model, weight_buffer_elem, state_dict=False):
    if state_dict:
        old_weights = []
        for w_name, new_w in zip([k for k in model.state_dict().keys() if 'alpha' not in k], weight_buffer_elem):
            path = w_name.split('.')
            old_weights.append(model.state_dict()[w_name])
            if len(path) > 1:
                f = operator.attrgetter('.'.join(path[:-1]))
                setattr(f(model), path[-1], new_w)
            else:
                setattr(model, w_name, new_w)
        return old_weights
    else:
        with torch.no_grad():
            old_weights = {w_name:w.clone() for w_name, w in model.named_weight_params()}
            for (w_name, w_old), (w_name_new, w_new) in zip(model.named_weight_params(), weight_buffer_elem.items()):
                # w_old.copy_(w_new)
                w_old.data = w_new
        return old_weights

def hinge_loss(x,y, threshold):
    if abs(x-y) <= threshold:
        return 0
    elif x > y:
        return x -y - threshold
    elif x < y:
        return y - x + threshold

def inverse_softplus(x, beta):
    return np.log(np.exp(beta*x)-1)/beta

def get_optimizers(model, config, grad = None):
    # Weight decay is realized only manually through compute_train_loss
    if type(config["w_lr"]) is float:
        true_w_lr = config["w_lr"]
    else:
        if not config["softplus_alpha_lr"]:
            true_w_lr = abs(config["w_lr"].item())
        else:
            true_w_lr= abs(F.softplus(config["w_lr"], config["softplus_beta"]).item())

    if config["w_optim"] == 'SGD':
        w_optimizer = SGD(model.weight_params(), lr=true_w_lr, momentum=config["w_momentum"])
    elif config ['w_optim'] =='Adam':
        w_optimizer = Adam(model.weight_params(), lr=true_w_lr)
    elif config["w_optim"] == "HyperSGD":
        w_optimizer = HyperSGD(model.weight_params(), lr=true_w_lr, momentum=config["w_momentum"], grad=grad)


    if config['w_scheduler'] == "step":
        w_scheduler = torch.optim.lr_scheduler.StepLR(w_optimizer, max(round(config["epochs"]/3), 1), gamma=0.1, verbose=False)
    elif config["w_scheduler"] is None:
        w_scheduler = None
    else:
        raise NotImplementedError

    if config["train_arch"]:
        if config['a_optim'] == 'SGD':
            a_optimizer = SGD(model.arch_params(), lr=config["a_lr"], momentum=config["a_momentum"])
        elif config['a_optim'] == 'Adam':
            a_optimizer = Adam(model.arch_params(), lr=config["a_lr"])
        
        if config["a_scheduler"] == 'step':
            a_scheduler = torch.optim.lr_scheduler.StepLR(a_optimizer, max(round(config["epochs"]/3), 1), gamma=0.1, verbose=False)
        elif config["a_scheduler"] is None:
            a_scheduler = None
        else:
            raise NotImplementedError
    else:
        # Placeholder optimizer that won't do anything - but the parameter list cannot be empty
        a_optimizer = None
        a_scheduler = None


    return {"w_optimizer":w_optimizer, "a_optimizer":a_optimizer, 
        "w_scheduler":w_scheduler, "a_scheduler":a_scheduler}

def vae_loss(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def get_criterion(model_type, dataset_cfg, preferred_loss=None):
    criterion=None
    if preferred_loss == "mse":
        criterion = torch.nn.MSELoss()
    elif preferred_loss == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    
    else:
        print("Trying to guess proper loss from model name")
        if model_type in ["MNIST", "log_regression", "MLP", "pt_logistic_l1", "log_reg"]:
            criterion = torch.nn.CrossEntropyLoss()
        elif model_type in ["max_deg", "softmax_mult", "linear", "fourier", "polynomial", "sigmoid", "AE", "linearAE"]:
            criterion = torch.nn.MSELoss()

    print(f"Using loss {criterion} for training")

    assert (not ('AE' in model_type and isinstance(criterion, torch.nn.CrossEntropyLoss)))
    assert (not (dataset_cfg['n_classes'] > 1 and isinstance(criterion, torch.nn.MSELoss) and not 'AE' in model_type) or preferred_loss=='mse')

    return criterion