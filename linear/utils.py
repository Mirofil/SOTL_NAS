import torch
from torch import Tensor
from torch.nn import Linear, MSELoss, functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt

def sample_tau(alpha,beta):
    return torch.tensor([np.random.gamma(alpha, 1/beta)])

def sample_C(alpha,beta):
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

def evidence_np(C, c, y, alpha_1, alpha_2, beta_1, beta_2, N, design_matrix):
    numerator = beta_1**alpha_1 + beta_2**alpha_2 * math.gamma(alpha_1 + N/2) / (2*math.pi)**(N/2)
    denominator = (2*math.pi)**(N/2) * math.gamma(alpha_1) * math.gamma(alpha_2)
    A = design_matrix.T @ design_matrix + C*c_tilde(c)
    matrix_term = np.linalg.det(A)**(1/2) * (beta_1 + 1/2*(y.T @ (np.identity(2*N-1) - design_matrix @ np.linalg.inv(A) @ design_matrix.T) @ y))**(-alpha_1-N/2)
    tail_constants = C**(len(c)+alpha_2-1/2) * np.exp(-beta_2*C) * c[0]**(1/2) * np.prod(c[1:])
    
    return numerator / denominator * matrix_term * tail_constants

def evidence(design_matrix, y, C, c, alpha_1, alpha_2, beta_1, beta_2, N):
    numerator = beta_1**alpha_1 * beta_2**alpha_2 * math.gamma(alpha_1 + N/2) / (2*math.pi)**(N/2)
    denominator = (2*math.pi)**(N/2) * math.gamma(alpha_1) * math.gamma(alpha_2)
    A = torch.add(design_matrix.T @ design_matrix, C*c_tilde(c))
    matrix_term = torch.det(A)**(1/2) * (beta_1 + 1/2*(y.T @ (torch.eye(N) - design_matrix @ torch.inverse(A) @ design_matrix.T) @ y))**(-alpha_1-N/2)
    tail_constants = C**(len(c)+alpha_2-1/2) * torch.exp(-beta_2*C) * c[0]**(1/2) * torch.prod(c[1:])
    
    return numerator / denominator * matrix_term * tail_constants

def evidence_log(design_matrix, y, C, c, alpha_1, alpha_2, beta_1, beta_2, N):
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

def featurize(x: float, max_order:int =4,  type:str ="fourier"):
    if type == 'fourier':
        # max_order should be 2*D when doing sum_{i=1}^D (Fourier terms), ie. each sin/cos term counts separately!
        featurized_input = [1]
        assert max_order % 2 == 0
        for order in range(0,max_order):
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
        
    return list(featurized_input)

def eval_features(x, max_order=2, type='fourier', noise_var=1):
    noise = np.random.normal(0, noise_var**(1/2))
    if type == "fourier":
        feature = np.array(x[:max_order]).sum()
    elif type == 'polynomial':
        feature = np.array(x[:max_order]).sum()
    
    return [feature+noise]

# define our data generation function
def data_generator(data_size=1000, max_order=5, max_order_y=None, noise_var=1, x_range=None, featurize_type='fourier', plot=False):
    inputs = []
    labels = []
    if max_order_y is None:
        max_order_y = max_order
    if x_range is None:
        x_range = 10*math.pi
    xs = np.linspace(-x_range,x_range,data_size)

    for x in xs:
        features = featurize(x, max_order=max_order, type=featurize_type)
        inputs.append(features)
        labels.append(eval_features(features, noise_var=noise_var, type=featurize_type))

    if plot:
        labels = [label[0] for label in labels]
        plt.plot(xs, labels)
    return inputs, labels

# def data_generator(data_size=1000, max_order=2, noise_var=1, featurize_type='fourier'):
#     inputs = []
#     labels = []

#     for i in range(data_size):
#         x = np.random.randint(2000) / 1000

#         y = (x * x) + (4 * x) - 3

#         features = featurize(x, max_order=max_order, type=featurize_type)
#         inputs.append(features)
#         labels.append(eval_features(features, noise_var=noise_var, type=featurize_type))

#     return inputs, labels

def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                      
def hessian(y, x1, x2):                                                                                    
    return jacobian(jacobian(y, x1, create_graph=True), x2)                                             
                                                                                                      