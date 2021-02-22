import torch
from layers import LinearSquash, LinearMaxDeg, FlexibleLinear, FeatureSelection
import torch.nn as nn
import torch.nn.functional as F
import itertools
from traits import FeatureSelectableTrait

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
    def __init__(self, num_features = 2, task='reg', model_type = "softmax_mult",
     weight_decay=0, n_classes=1, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.task = task
        self.num_features = num_features
        self.n_classes = n_classes
        if task == 'reg':
            assert n_classes == 1
        if model_type == "softmax_mult":
            self.fc1 = LinearSquash(num_features, n_classes, bias=False, squash_type="softmax", **kwargs)
            self.model = self.fc1
        elif model_type == "sigmoid":
            self.fc1 = LinearSquash(num_features, n_classes, bias=False, squash_type="sigmoid", **kwargs)
            self.alpha_feature_selectors = self.fc1.alphas
            self.feature_normalizers = self.fc1.weight
            self.model = self.fc1
        elif model_type == "max_deg":
            self.fc1 = LinearMaxDeg(num_features, n_classes, bias=False, **kwargs)

            self.model = self.fc1
        elif model_type == "linear":
            self.fc1 = FlexibleLinear(num_features, n_classes, bias=False)
            self.model = self.fc1
        elif model_type == "MNIST" or model_type == "MLP":
            self.model = MLP(input_dim=num_features,hidden_dim=1000,output_dim=n_classes, weight_decay=weight_decay)
        elif model_type == "log_regression":
            self.model = LogReg(input_dim=num_features, output_dim=n_classes)
        elif model_type == "AE":
            self.model = AE(input_dim=num_features)
        else:
            raise NotImplementedError
        self.alphas = []
        self.feature_indices = None
        self.model.feature_indices = self.feature_indices
        if weight_decay > 0:
            self.alpha_weight_decay = torch.nn.Parameter(torch.tensor([weight_decay], dtype=torch.float32, requires_grad=True).unsqueeze(dim=0))
        else:
            self.alpha_weight_decay = torch.tensor(0)
    def forward(self, x, weight=None, alphas=None, feature_indices=None):
        x = x.view(-1, self.num_features)
        if feature_indices is not None:
            for to_delete in range(x.shape[1]):
                if to_delete not in feature_indices:
                    x[:, to_delete] = 0 
        elif self.feature_indices is not None:
            for to_delete in range(x.shape[1]):
                if to_delete not in self.feature_indices:
                    x[:, to_delete] = 0 
        return self.model(x, weight, alphas)

    def adaptive_weight_decay(self):
        return torch.sum(torch.abs(self.fc1.weight*self.fc1.compute_deg_constants()))
        
    def set_features(self, feature_indices):
        self.feature_indices = set(feature_indices)
        self.model.feature_indices = feature_indices
    
    def alpha_feature_selectors(self):
        return self.model.alpha_feature_selectors()
    
    def feature_normalizers(self):
        return self.model.feature_normalizers()

        
class LogReg(nn.Module):
    def __init__(self, input_dim=28*28, output_dim=10):
        super(LogReg, self).__init__()
        self._input_dim = input_dim
        self.lin1 = nn.Linear(input_dim, output_dim)

    def forward(self, x, weights=None, alphas=None):
        x = x.view(-1, self._input_dim)
        x = self.lin1(x)
        return x

class MLP(RegressionNet, FeatureSelectableTrait):
    def __init__(self, input_dim=28*28, hidden_dim=1000, output_dim=10, weight_decay=0, **kwargs):
        super(MLP, self).__init__()
        self._input_dim = input_dim
        self.feature_selection = FeatureSelection(input_dim)
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, weights=None, alphas=None):

        x = x.view(-1, self._input_dim)
        x = self.feature_selection(x, feature_indices=self.feature_indices)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x
    
    def squash(self, *args, **kwargs):
        return self.feature_selection.squash(*args, **kwargs)

    def alpha_feature_selectors(self):
        return self.feature_selection.alphas
    
    def feature_normalizers(self):
        return self.feature_selection.weight

class AE(RegressionNet, FeatureSelectableTrait):
    def __init__(self, input_dim=28*28, **kwargs):
        super().__init__()
        self._input_dim = input_dim
        self.feature_selection = FeatureSelection(input_dim)
        self.encoder_hidden_layer = nn.Linear(
            in_features=input_dim, out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=input_dim
        )

    def forward(self, x, *args, **kwargs):
        x = x.view(-1, self._input_dim)
        x = self.feature_selection(x, feature_indices=self.feature_indices)
        activation = self.encoder_hidden_layer(x)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed
    def squash(self, *args, **kwargs):
        return self.feature_selection.squash(*args, **kwargs)
    def alpha_feature_selectors(self):
        return self.feature_selection.alphas
    
    def feature_normalizers(self):
        return self.feature_selection.weight