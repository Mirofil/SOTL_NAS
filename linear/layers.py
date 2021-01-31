import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

class Linear2(torch.nn.Linear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alphas = torch.nn.Parameter(torch.ones(1, self.in_features))
        # self.alphas = torch.nn.Parameter(torch.tensor([-0.0499, -0.0443,  0.1992]))
        # weights from a training run with D=3 - tensor([[-0.0499, -0.0443,  0.1992]]

    def forward(self, input: Tensor, weight: Tensor = None, alphas: Tensor = None) -> Tensor:
        if weight is None:
            weight = self.weight
        if alphas is None:
            alphas = self.alphas
        return F.linear(input, weight*F.softmax(alphas), self.bias)