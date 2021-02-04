import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

class Linear2(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias, **kwargs) -> None:
        super().__init__(in_features,out_features,bias)
        self.alphas = torch.nn.Parameter(torch.ones(1, in_features))

    def forward(self, input: Tensor, weight: Tensor = None, alphas: Tensor = None) -> Tensor:
        if weight is None:
            weight = self.weight
        else:
            weight = weight[0]
        if alphas is None:
            alphas = self.alphas
        return F.linear(input, weight*F.softmax(alphas), self.bias)

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
        self.alpha_constants = torch.tensor(constants,dtype=torch.float32).unsqueeze(dim=0)

    def forward(self, input: Tensor, weight: Tensor = None, alphas: Tensor = None) -> Tensor:
        if weight is None:
            weight = self.weight
        else:
            weight = weight[0]
        if alphas is None:
            alphas = self.alphas
        return F.linear(input, weight*self.squished_tanh(alphas+self.alpha_constants).to(self.alpha_constants.device), self.bias)

    @staticmethod
    def squished_tanh(x, plot=False):
        if plot:
            xs = np.linspace(-5,5,100)
            ys = [(F.tanh(4*torch.tensor(elem))+1)/2 for elem in xs]
            plt.plot(xs,ys)

        return (F.tanh(4*x)+1)/2

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
