import torch
from layers import Linear2, LinearMaxDeg

class Net(torch.nn.Module):
    def __init__(self, num_features = 2):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(num_features, 1, bias=False)

    def forward(self, x):
        return self.fc1(x)
    
    def weight_params(self):
        for p in [self.fc1.weight]:
            yield p
    
    def arch_params(self):
        for p in [self.fc1.alphas]:
            yield p

class SoTLNet(Net):
    def __init__(self, num_features = 2, layer_type = "softmax_mult", **kwargs):
        super().__init__()
        if layer_type == "softmax_mult":
            self.fc1 = Linear2(num_features, 1, bias=False, **kwargs)
        elif layer_type == "max_deg":
            self.fc1 = LinearMaxDeg(num_features, 1, bias=False, **kwargs)

    def forward(self, x, weight=None, alphas=None):
        return self.fc1(x, weight, alphas)
