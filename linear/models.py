import torch
from layers import Linear2, LinearMaxDeg, FlexibleLinear
import torch.nn as nn
import torch.nn.functional as F
class RegressionNet(torch.nn.Module):
    def __init__(self, num_features = 2, **kwargs):
        super(RegressionNet, self).__init__()
        self.alphas = []

    def forward(self, x):
        return self.fc1(x)
    
    def weight_params(self):
        for n,p in self.named_parameters():
            if 'alpha' not in n:
                yield p
            else:
                continue
    
    def arch_params(self):
        for n,p in self.named_parameters():
            if 'alpha' in n:
                yield p
            else:
                continue

class SoTLNet(RegressionNet):
    def __init__(self, num_features = 2, model_type = "softmax_mult", weight_decay=0, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        if model_type == "softmax_mult":
            self.fc1 = Linear2(num_features, 1, bias=False, **kwargs)
            self.model = self.fc1
        elif model_type == "max_deg":
            self.fc1 = LinearMaxDeg(num_features, 1, bias=False, **kwargs)
            self.model = self.fc1
        elif model_type == "linear":
            self.fc1 = FlexibleLinear(num_features, 1, bias=False)
            self.model = self.fc1
        elif model_type == "MNIST":
            self.model = MLP(input_dim=28*28,hidden_dim=1000,output_dim=10)
        elif model_type == "log_regression":
            self.model = LogReg(input_dim=28*28, output_dim=10)
        else:
            raise NotImplementedError
        self.alphas = []
        if weight_decay > 0:
            self.alpha_weight_decay = torch.nn.Parameter(torch.tensor([weight_decay], dtype=torch.float32, requires_grad=True).unsqueeze(dim=0))
            self.alphas.append(self.alpha_weight_decay)
        else:
            self.alpha_weight_decay = 0
    def forward(self, x, weight=None, alphas=None):
        return self.model(x, weight, alphas)
        
class LogReg(nn.Module):
    def __init__(self, input_dim=28*28, output_dim=10):
        super(LogReg, self).__init__()
        self._input_dim = input_dim
        self.lin1 = nn.Linear(input_dim, output_dim)

    def forward(self, x, weights=None, alphas=None):
        x = x.view(-1, self._input_dim)
        x = self.lin1(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=1000, output_dim=10):
        super(MLP, self).__init__()
        self._input_dim = input_dim
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, weights=None, alphas=None):
        x = x.view(-1, self._input_dim)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
