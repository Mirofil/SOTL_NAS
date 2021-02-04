import torch
import torch.nn as nn

def get_criterion(model_type):
    criterion=None
    if model_type in ["MNIST", "log_regression"]:
        criterion = torch.nn.CrossEntropyLoss()
    elif model_type in ["max_deg", "softmax_mult", "linear", "fourier", "polynomial"]:
        criterion = torch.nn.MSELoss()
    
    return criterion
