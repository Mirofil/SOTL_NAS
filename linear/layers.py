import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from traits import FeatureSelectableTrait


class LinearSquash(torch.nn.Linear, FeatureSelectableTrait):
    def __init__(self, in_features, out_features, bias, squash_type="softmax", **kwargs) -> None:
        super().__init__(in_features,out_features,bias)
        self.alphas = torch.nn.Parameter(torch.zeros(1, in_features))
        self.squash_type = squash_type


    def forward(self, input: Tensor, weight: Tensor = None, alphas: Tensor = None) -> Tensor:
        if weight is None:
            weight = self.weight
        else:
            weight = weight[0]
        if alphas is None:
            alphas = self.alphas
        return F.linear(input, weight*self.squash(alphas), self.bias)
    
    def squash_constants(self):
        return self.squash(self.alphas)
    
    def squash(self, *args, **kwargs):
        if self.squash_type == "softmax":
            return F.softmax
        elif self.squash_type == "sigmoid":
            return torch.sigmoid

    def alpha_feature_selectors(self):
        return self.alphas
    
    def feature_normalizers(self):
        return self.weight

class FeatureSelection(torch.nn.Module, FeatureSelectableTrait):
    def __init__(self, in_features, squash_type="sigmoid", **kwargs) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((1,in_features), requires_grad=False))
        self.feature_indices = {i for i in range(in_features)}

        self.alphas = torch.nn.Parameter(torch.zeros(1, in_features))
        self.squash_type = squash_type

    def forward(self, x: Tensor, feature_indices=None) -> Tensor:
        if feature_indices is not None:
            for to_delete in range(x.shape[1]):
                if to_delete not in feature_indices:
                    x[:, to_delete] = 0 
        elif self.feature_indices is not None:
            for to_delete in range(x.shape[1]):
                if to_delete not in self.feature_indices:
                    x[:, to_delete] = 0 
        return x * self.squash(self.alphas)

    def squash(self, *args, **kwargs):
        if self.squash_type == "softmax":
            return F.softmax(*args, **kwargs)
        elif self.squash_type == "sigmoid":
            return torch.sigmoid(*args, **kwargs)

class LinearMaxDeg(torch.nn.Linear):
    def __init__(self, *args, degree=30, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alphas = torch.nn.Parameter(torch.tensor([degree], dtype=torch.float32).unsqueeze(dim=0))
        constants = [0]
        for i in range(1,self.in_features+1):
            constants.append(-i*1) # the multiplicative constant here depends on the width of the soft step function used
            constants.append(-i*1)
        constants = constants[:self.in_features]
        self.degree = self.alphas
        self.alpha_constants = torch.nn.Parameter(torch.tensor(constants,dtype=torch.float32).unsqueeze(dim=0), requires_grad=False)

    def forward(self, input: Tensor, weight: Tensor = None, alphas: Tensor = None) -> Tensor:
        if weight is None:
            weight = self.weight
        else:
            weight = weight[0]
        if alphas is None:
            alphas = self.alphas
        else:
            alphas = alphas[0]

        return F.linear(input, weight*self.compute_deg_constants(alphas=alphas).to(self.alpha_constants.device), self.bias)

    def compute_deg_constants(self, alphas = None):
        if alphas is None:
            alphas = self.degree
        return self.squished_tanh(alphas+self.alpha_constants)

    @staticmethod
    def squished_tanh(x, plot=False):
        if plot:
            xs = np.linspace(-5,5,100)
            ys = [(F.tanh(1*torch.tensor(elem))+1)/2 for elem in xs]
            plt.plot(xs,ys)

        return (F.tanh(x)+torch.tensor(1))/2

class FlexibleLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias=False)
        self.alphas = None

    def forward(self, input: Tensor, weight: Tensor = None, alphas: Tensor = None, **kwargs) -> Tensor:
        if weight is None:
            weight = self.weight
        else:
            weight = weight[0]

        return F.linear(input, weight, self.bias)
