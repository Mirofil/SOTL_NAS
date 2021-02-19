import torch
from layers import LinearSquash, LinearMaxDeg, FlexibleLinear
import torch.nn as nn
import torch.nn.functional as F
class RegressionNet(torch.nn.Module):
    def __init__(self, num_features = 2, **kwargs):
        super(RegressionNet, self).__init__()
        self.alphas = []

    def forward(self, x):
        return self.fc1(x)

    def named_weight_params(self):
        for n,p in self.named_parameters():
            if 'alpha' not in n:
                yield (n,p)
            else:
                continue
    
    def weight_params(self):
        for n,p in self.named_parameters():
            if 'alpha' not in n:
                yield p
            else:
                continue
    
    def arch_params(self):
        for n,p in self.named_parameters():
            if 'alpha' in n and p.requires_grad:
                yield p
            else:
                continue

class SoTLNet(RegressionNet):
    def __init__(self, num_features = 2, model_type = "softmax_mult", weight_decay=0, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        if model_type == "softmax_mult":
            self.fc1 = LinearSquash(num_features, 1, bias=False, squash_type="softmax", **kwargs)
            self.model = self.fc1
        elif model_type == "sigmoid":
            self.fc1 = LinearSquash(num_features, 1, bias=False, squash_type="sigmoid", **kwargs)
            self.model = self.fc1
        elif model_type == "max_deg":
            self.fc1 = LinearMaxDeg(num_features, 1, bias=False, **kwargs)

            self.model = self.fc1
        elif model_type == "linear":
            self.fc1 = FlexibleLinear(num_features, 1, bias=False)
            self.model = self.fc1
        elif model_type == "MNIST" or model_type == "MLP":
            self.model = MLP(input_dim=28*28,hidden_dim=1000,output_dim=10, weight_decay=weight_decay)
        elif model_type == "log_regression":
            self.model = LogReg(input_dim=28*28, output_dim=10)
        else:
            raise NotImplementedError
        self.alphas = []
        if weight_decay > 0:
            self.alpha_weight_decay = torch.nn.Parameter(torch.tensor([weight_decay], dtype=torch.float32, requires_grad=True).unsqueeze(dim=0))
            # self.alphas.append(self.alpha_weight_decay)
        else:
            self.alpha_weight_decay = torch.tensor(0)
    def forward(self, x, weight=None, alphas=None):
        return self.model(x, weight, alphas)

    def adaptive_weight_decay(self):
        return torch.sum(torch.abs(self.fc1.weight*self.fc1.compute_deg_constants()))
        
class LogReg(nn.Module):
    def __init__(self, input_dim=28*28, output_dim=10):
        super(LogReg, self).__init__()
        self._input_dim = input_dim
        self.lin1 = nn.Linear(input_dim, output_dim)

    def forward(self, x, weights=None, alphas=None):
        x = x.view(-1, self._input_dim)
        x = self.lin1(x)
        return x

class MLP(RegressionNet):
    def __init__(self, input_dim=28*28, hidden_dim=1000, output_dim=10, weight_decay=0, **kwargs):
        super(MLP, self).__init__()
        self._input_dim = input_dim
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        # self.lin22 = nn.Linear(hidden_dim, hidden_dim)
        # self.lin23 = nn.Linear(hidden_dim, hidden_dim)

        self.lin3 = nn.Linear(hidden_dim, output_dim)
        # if weight_decay > 0:
        #     self.alpha_weight_decay = torch.nn.Parameter(torch.tensor([weight_decay], dtype=torch.float32, requires_grad=True).unsqueeze(dim=0))
        # else:
        #     self.alpha_weight_decay = 0

    def forward(self, x, weights=None, alphas=None):

        # with torch.no_grad():
        #     if weights is not None:
        #         old_weights = [w.clone() for w in self.weight_params()]

        #         for w_old, w_new in zip(self.weight_params(), weights):
        #             w_old.copy_(w_new)



        x = x.view(-1, self._input_dim)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        # x = F.relu(self.lin22(x))
        # x = F.relu(self.lin23(x))


        x = self.lin3(x)

        # with torch.no_grad():
        #     if weights is not None:
        #         for w_old, w_new in zip(self.weight_params(), old_weights):
        #             w_old.copy_(w_new)
        return x
