import torch
from layers import Linear2, LinearMaxDeg

class RegressionNet(torch.nn.Module):
    def __init__(self, num_features = 2, **kwargs):
        super(RegressionNet, self).__init__()
        self.fc1 = torch.nn.Linear(num_features, 1, bias=False)
        self.alphas = []

    def forward(self, x):
        return self.fc1(x)
    
    def weight_params(self):
        for p in [self.fc1.weight]:
            yield p
    
    def arch_params(self):
        for p in [self.fc1.alphas] + self.alphas:
            yield p

class SoTLNet(RegressionNet):
    def __init__(self, num_features = 2, layer_type = "softmax_mult", weight_decay=0, **kwargs):
        super().__init__(**kwargs)
        if layer_type == "softmax_mult":
            self.fc1 = Linear2(num_features, 1, bias=False, **kwargs)
        elif layer_type == "max_deg":
            self.fc1 = LinearMaxDeg(num_features, 1, bias=False, **kwargs)
        self.alphas = []
        if weight_decay > 0:
            self.weight_decay = torch.nn.Parameter(torch.tensor([weight_decay], dtype=torch.float32, requires_grad=True).unsqueeze(dim=0))
            self.alphas = self.alphas.append(self.weight_decay)
    def forward(self, x, weight=None, alphas=None):
        return self.fc1(x, weight, alphas)
