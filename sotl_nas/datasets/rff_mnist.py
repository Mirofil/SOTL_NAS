import torch
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from torchvision import transforms, datasets
import pickle as pkl
# from linear_regression import *
# from optimize_then_prune import BLRModel, optimize_linear_combo 
from tqdm import tqdm
import pandas as pd


def build_embedding(d, k=None, l=None):

    w = 1/np.sqrt(l) * torch.rand(d, k).cuda()
    b = 2*np.pi * np.random.rand(d)

    def f(X):
        n = X.shape[0]
        X = torch.tensor(X, dtype=torch.float32).cuda()
        fs = (w @ X.T) .T + torch.tensor(np.repeat([b], n, axis=0)).cuda()
        return np.sqrt(2/d) * torch.cos(fs)
    return f

def get_MNIST(root="./"):
    input_size = 28
    channels = 1
    num_classes = 10
    fmnist = datasets.MNIST(root + "data/MNIST", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
    data_loader = torch.utils.data.DataLoader(fmnist,
                                          batch_size=10,
                                          shuffle=True)
    ftest = datasets.MNIST(root + "data/MNIST", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

    return input_size, num_classes, fmnist, ftest

def load_subset_mnist(max_label=2, max_size=1000, reshape=True):
    root = '.'
    input_size = 28
    channels = 1
    num_classes = 10
    k = max_size
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.LinearTransformation(torch.eye(input_size**2), torch.zeros(input_size**2))
    ])
    mnist = datasets.MNIST(root + "data/MNIST", train=True, transform=input_transform, target_transform=None, download=True)
    
    x = mnist.data.reshape(-1, input_size**2)
    y = mnist.targets.numpy().reshape(-1)
    test = datasets.MNIST(root + "data/MNIST", train=False, transform=input_transform, target_transform=None, download=True)
    xtest = test.data.reshape(-1, input_size**2)
    ytest = test.targets.numpy().reshape(-1)
    x = x.numpy()

    # Subset generation

    x = x[np.where(y <max_label)]#[:k]
    y = y[np.where(y < max_label)]#[:k]
    x = x[:k]
    y = y[:k]
    xtest= xtest.numpy()
    xtest = xtest[np.where(ytest <max_label)]
    ytest = ytest[np.where(ytest < max_label)]

    return xtest, ytest, x, y

def generate_rff(d, X, l=1.0):    
    if len(X.shape)==1:
        X = X.reshape([1, -1])
    X = torch.tensor(X, dtype=torch.float32).cuda()
    k = X.shape[1]
    n = X.shape[0]
    # sample biases uniformly from 0, 2pi
    b = 2*np.pi * np.random.rand(d)
    # Sample w according to normal(0, 1/l**2)
    W = 1/np.sqrt(l) * torch.rand(d, k).cuda()
    print('types', type(X), type(W))
    fs = (W @ X.T) .T + torch.tensor(np.repeat([b], n, axis=0)).cuda()
    return np.sqrt(2/d) * torch.cos(fs) #np.bmat([np.cos(fs), np.sin(fs)])

def generate_rff_mnist(lengthscale, fourier_dim, max_label=2):
    xtest, ytest, x, y = load_subset_mnist(max_label)
    n = len(y)
    # fx = l(fourier_dim, np.concatenate([x, xtest], axis=0))

    # Want to use same random fourier features to produce train and test set
    fx = generate_rff(fourier_dim, np.concatenate([x, xtest], axis=0), l=lengthscale)   
    ftest = fx[n:]
    fx = fx[:n]
    return ftest, ytest, fx, y