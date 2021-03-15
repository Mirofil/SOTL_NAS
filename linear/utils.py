import itertools
import math
import random
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, MSELoss
from torch.nn import functional as F
from tqdm import tqdm


def prepare_seed(rand_seed):
  random.seed(rand_seed)
  np.random.seed(rand_seed)
  torch.manual_seed(rand_seed)
  torch.cuda.manual_seed(rand_seed)
  torch.cuda.manual_seed_all(rand_seed)

def sample_tau(alpha,beta) -> Tensor:
    return torch.tensor([np.random.gamma(alpha, 1/beta)])

def sample_C(alpha,beta) -> Tensor:
    return torch.tensor([np.random.gamma(alpha, 1/beta)])

def c_cov(c, as_tensor=True):
    cov = np.identity(len(c))
    for i, elem in enumerate(c):
        cov[i][i] = c[i]

    if as_tensor:
        return torch.tensor(cov)
    else:
        return cov

def c_tilde(c, as_tensor=True):
    c_duplicated = [[elem, elem] for elem in c]
    c_duplicated = list(itertools.chain.from_iterable(c_duplicated))
    c_duplicated = c_duplicated[1:]
    cov = np.identity(len(c_duplicated))
    for i, elem in enumerate(c_duplicated):
        cov[i][i] = c_duplicated[i]
    if as_tensor:
        return torch.tensor(cov)
    else:
        return cov

def sample_w(C, tau, c, as_tensor=True):
    cov = c_cov(c)
    cov = C * tau * cov
    if as_tensor:
        return torch.tensor(np.random.multivariate_normal(mean=[0 for _ in range(len(c))], cov=cov))
    else:
        return np.random.multivariate_normal(mean=[0 for _ in range(len(c))], cov=cov)

def evidence_np(C, c, y, alpha_1, alpha_2, beta_1, beta_2, N, design_matrix) -> float:
    numerator = beta_1**alpha_1 + beta_2**alpha_2 * math.gamma(alpha_1 + N/2) / (2*math.pi)**(N/2)
    denominator = (2*math.pi)**(N/2) * math.gamma(alpha_1) * math.gamma(alpha_2)
    A = design_matrix.T @ design_matrix + C*c_tilde(c)
    matrix_term = np.linalg.det(A)**(1/2) * (beta_1 + 1/2*(y.T @ (np.identity(2*N-1) - design_matrix @ np.linalg.inv(A) @ design_matrix.T) @ y))**(-alpha_1-N/2)
    tail_constants = C**(len(c)+alpha_2-1/2) * np.exp(-beta_2*C) * c[0]**(1/2) * np.prod(c[1:])
    
    return numerator / denominator * matrix_term * tail_constants

def evidence(design_matrix, y, C, c, alpha_1, alpha_2, beta_1, beta_2, N) -> float:
    numerator = beta_1**alpha_1 * beta_2**alpha_2 * math.gamma(alpha_1 + N/2) / (2*math.pi)**(N/2)
    denominator = (2*math.pi)**(N/2) * math.gamma(alpha_1) * math.gamma(alpha_2)
    A = torch.add(design_matrix.T @ design_matrix, C*c_tilde(c))
    matrix_term = torch.det(A)**(1/2) * (beta_1 + 1/2*(y.T @ (torch.eye(N) - design_matrix @ torch.inverse(A) @ design_matrix.T) @ y))**(-alpha_1-N/2)
    tail_constants = C**(len(c)+alpha_2-1/2) * torch.exp(-beta_2*C) * c[0]**(1/2) * torch.prod(c[1:])
    
    return numerator / denominator * matrix_term * tail_constants

def evidence_log(design_matrix, y, C, c, alpha_1, alpha_2, beta_1, beta_2, N) -> float:
    # numerator = beta_1**alpha_1 * beta_2**alpha_2 * math.gamma(alpha_1 + N/2) / (2*math.pi)**(N/2)
    numerator = alpha_1*torch.log(beta_1) + alpha_2*torch.log(beta_2) + torch.lgamma(alpha_1+N/2) - N/2*torch.log(2*torch.tensor([math.pi]))
    # denominator = (2*math.pi)**(N/2) * math.gamma(alpha_1) * math.gamma(alpha_2)
    denominator = N/2 * torch.log(2*torch.tensor([math.pi]))+ torch.lgamma(alpha_1) + torch.lgamma(alpha_2)

    A = torch.add(design_matrix.T @ design_matrix, C*c_tilde(c))
    # matrix_term = torch.det(A)**(1/2) * (beta_1 + 1/2*(y.T @ (torch.eye(2*D-1) - design_matrix @ torch.inverse(A) @ design_matrix.T) @ y))**(-alpha_1-N/2)
    matrix_term = 1/2*torch.log(torch.det(A)) + (-alpha_1-N/2)*torch.log((beta_1 + 1/2*(y.T @ (torch.eye(N) - design_matrix @ torch.inverse(A).type(torch.FloatTensor) @ design_matrix.T) @ y)))
    
    # tail_constants = C**(len(c)+alpha_2-1/2) * torch.exp(-beta_2*C) * c[0]**(1/2) * torch.prod(c[1:])
    tail_constants = (len(c) + alpha_2 - 1/2)*torch.log(C) + (-beta_2*C) + 1/2*torch.log(c[0]) + torch.sum(c[1:])
    return numerator - denominator + matrix_term + tail_constants

def featurize(x: float, max_order:int =4, type:str ="fourier") -> Sequence:
    featurized_input = None
    if type == 'fourier':
        # max_order should be 2*D when doing sum_{i=1}^D (Fourier terms), ie. each sin/cos term counts separately!
        featurized_input = [1]
        assert max_order % 2 == 0
        for order in range(1,max_order):
            featurized_input.append(np.cos(order*x))
            featurized_input.append(np.sin(order*x))
        featurized_input = featurized_input[0:max_order+1]
    elif type == "polynomial":
        features = [lambda x: np.power(x, 0),
            lambda x: np.power(x, 1),
            lambda x: np.power(x, 2),
            lambda x: np.power(x, 3),
            lambda x: np.power(x, 4),
            lambda x: np.power(x, 5),
            lambda x: np.power(x, 6),
            lambda x: np.power(x, 7),
            lambda x: np.power(x, 8),
            lambda x: np.power(x, 9),
            lambda x: np.power(x, 10),
            lambda x: np.sin(2 * np.pi * x),
            lambda x: np.cos(2 * np.pi * x),
            lambda x: np.exp(x)]
    
        featurized_input = []
        for feature in features[:max_order]:
            featurized_input.append(feature(x))
    else:
        raise NotImplementedError
        
    return list(featurized_input)

def eval_features(x:Sequence, max_order:int=2, type:str='fourier', noise_var:float=1, summation='raw') -> Sequence:
    noise = np.random.normal(0, noise_var**(1/2))
    final_features = None
    feature = None

    if isinstance(max_order, int):
        final_features = x[:max_order]
        if type == "fourier":
            assert (max_order % 2 == 1) or max_order == 1 #The constant term is there once only, then each degree of Fourier basis comes in pairs

    elif isinstance(max_order, (tuple, list)):
        final_features = []
        for seq in max_order:
            final_features = final_features + x[seq[0]:seq[1]]

    if summation == 'gaussian':
        weights = random.normal(loc=1, scale=1, size = len(final_features))
    else:
        weights = np.ones(shape = len(final_features))


    feature = (np.array(final_features) * weights).sum()

    return {"features":[feature+noise], "weights": weights}

def data_generator(n_samples:int=1000, n_features:int=5, n_informative:int=None, max_order_x:int=None, 
    noise:float=1, x_range:float=None, featurize_type:str='fourier', plot:bool=False, shuffle_features=False):
    inputs = []
    labels = []
    if n_informative is None:
        n_informative = n_features
    if max_order_x is None:
        max_order_x = n_features
    if x_range is None:
        if featurize_type == "fourier":
            x_range = 10000*math.pi
        else:
            x_range = 10
    xs = np.linspace(-x_range,x_range,n_samples)

    for x in tqdm(xs, desc = f"Generating {featurize_type} features"):
        final_features = None
        features = featurize(x, max_order=n_features, type=featurize_type)
        evaled_features = eval_features(features, noise_var=noise, max_order=n_informative, type=featurize_type)
        labels.append(evaled_features["features"])
        if isinstance(max_order_x, int):
            final_features = features[:max_order_x]
        elif isinstance(max_order_x, (tuple, list)):
            final_features = []
            for seq in max_order_x:
                final_features = final_features + features[seq[0]:seq[1]]
        else:
            raise NotImplementedError

        inputs.append(final_features)

    if shuffle_features:
        perm = np.random.permutation(range(len(inputs[0])))
        inputs = [[x[i] for i in perm] for x in inputs]

    if plot:
        labels = [label[0] for label in labels]
        plt.plot(xs, labels)
    
    print(f"Dataset generated with weights: {evaled_features['weights']}")


    return inputs, labels


def jacobian(y: Tensor, x: Tensor, create_graph:bool=False) -> Tensor:                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph, allow_unused=True)
        # print(grad_x)
        if grad_x is not None:
            jac.append(grad_x.reshape(x.shape))
        else:
            jac.append(torch.zeros(x.shape))                                                           
        grad_y[i] = 0.               
    # print(jac)

    final_shape = (1, x.shape[1]) if len(y.shape) == 0 else (y.shape[1], min(x.shape[1], y.shape[1]))
    # print(final_shape)                 
    # print(x)                               
    return torch.stack(jac).reshape(final_shape)                                                
                                                                                                      
def hessian(y:Tensor, x1:Tensor, x2:Tensor) -> Tensor:                                                                                    
    return jacobian(jacobian(y, x1, create_graph=True), x2)                                             
                                                                                                      