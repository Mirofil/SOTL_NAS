import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

class Linear2(torch.nn.Linear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alphas = torch.nn.Parameter(torch.ones(1, self.in_features))

    def forward(self, input: Tensor, weight: Tensor = None, alphas: Tensor = None) -> Tensor:
        if weight is None:
            weight = self.weight
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
        self.cs = torch.tensor(constants,dtype=torch.float32).unsqueeze(dim=0)

    def forward(self, input: Tensor, weight: Tensor = None, alphas: Tensor = None) -> Tensor:
        if weight is None:
            weight = self.weight
        if alphas is None:
            alphas = self.alphas
        return F.linear(input, weight*self.squished_tanh(self.alphas+self.cs), self.bias)

    @staticmethod
    def squished_tanh(x, plot=False):
        if plot:
            xs = np.linspace(-5,5,100)
            ys = [(F.tanh(4*torch.tensor(elem))+1)/2 for elem in xs]
            plt.plot(xs,ys)

        return (F.tanh(4*x)+1)/2

class LinearRidge(torch.nn.Linear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alphas = torch.nn.Parameter(torch.ones(1, self.in_features))

    def forward(self, input: Tensor, weight: Tensor = None, alphas: Tensor = None) -> Tensor:
        if weight is None:
            weight = self.weight
        if alphas is None:
            alphas = self.alphas
        return F.linear(input, weight*F.softmax(alphas), self.bias)
