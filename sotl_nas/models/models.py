import itertools
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sotl_nas.models.base import Hypertrainable
from sotl_nas.models.layers import (EmbeddingCombiner, FeatureSelection,
                                    HyperBatchNorm2d, HyperConv2d, HyperLinear,
                                    LinearMaxDeg, LinearSquash, RFFEmbedding,
                                    Supernetwork)
from sotl_nas.models.traits import (AutoEncoder, Classifier,
                                    FeatureSelectableTrait, Regressor)
from sotl_nas.utils.train import record_parents, switch_weights


class SoTLNet(Hypertrainable):
    def __init__(self, n_features = None, model_type = "softmax_mult", task="whatever",
     alpha_weight_decay=0, n_classes=1, cfg=None, alpha_lr=None, alpha_w_momentum=None, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.n_features = n_features
        self.n_classes = n_classes
        self.cfg = cfg


        if model_type == "softmax_mult":
            self.fc1 = LinearSquash(n_features, n_classes, bias=False, squash_type="softmax", **kwargs)
            self.model = self.fc1
        elif model_type == "sigmoid":
            self.fc1 = LinearSquash(n_features, n_classes, bias=False, squash_type="sigmoid", **kwargs)

            self.model = self.fc1
        elif model_type == "rff":
            l = cfg["l"] if "l" in cfg.keys() else 1
            self.model = RFFRegression(1000, n_features, l)
        elif model_type == "rff_bag":
            self.model = RFFRegressionBag(1000, n_features)
        elif model_type == "max_deg":
            self.fc1 = LinearMaxDeg(n_features, n_classes, bias=False, degree=cfg["initial_degree"], **kwargs)

            self.model = self.fc1
        elif model_type == "linear":
            self.fc1 = HyperLinear(n_features, n_classes, bias=False)
            self.model = self.fc1
        elif model_type == "LinearSupernetRFF":
            self.model = LinearSupernetworkRFF(input_dim=n_features, output_dim = n_classes)
        elif model_type == "MNIST" or model_type == "MLP":
            self.model = MLP(input_dim=n_features,hidden_dim=1000,output_dim=n_classes)
        elif model_type == "MLP2":
            self.model = MLP2(input_dim=n_features,hidden_dim=1000,output_dim=n_classes)
        elif model_type == "MLP_sigmoid_relu":
            self.model = MLP_sigmoid_relu(input_dim=n_features,hidden_dim=1000,output_dim=n_classes)
        elif model_type == "MLPLarge":
            self.model = MLPLarge(input_dim=n_features,hidden_dim=1000,output_dim=n_classes)
        elif model_type =="vgg":
            self.model = VGG(make_layers(vgg_cfg['D'], batch_norm=True))
        elif model_type == "pt_logistic_l1":
            self.model = LogReg(input_dim=n_features, output_dim=n_classes)
            print("Setting (overriding?) default decay values for logistic L1")
            self.config["w_decay_order"] = 1
            self.config["w_weight_decay"] = 1
            self.config["a_decay_order"] = 0
        elif model_type == "log_reg":
            self.model = LogReg(input_dim=n_features, output_dim=n_classes)
        elif model_type == "AE":
            self.model = AE(input_dim=n_features)
        elif model_type == "linearAE":
            self.model = LinearAE(input_dim=n_features)
        else:
            raise NotImplementedError
        self.alphas = []
        self.feature_indices = None
        self.model.feature_indices = self.feature_indices
        if alpha_weight_decay is not None and alpha_weight_decay > 0:
            self.alpha_weight_decay = torch.nn.Parameter(torch.tensor(alpha_weight_decay, dtype=torch.float32, requires_grad=True).unsqueeze(dim=0))
        else:
            self.alpha_weight_decay = torch.tensor(0)
        if alpha_lr is not None and alpha_lr > 0:
            self.alpha_lr = torch.nn.Parameter(torch.tensor(alpha_lr, dtype=torch.float32, requires_grad=True).unsqueeze(dim=0))
        else:
            self.alpha_lr = torch.tensor(0)
        if alpha_w_momentum is not None and alpha_w_momentum > 0:
            self.alpha_w_momentum = torch.nn.Parameter(torch.tensor(alpha_w_momentum, dtype=torch.float32, requires_grad=True).unsqueeze(dim=0))
        else:
            self.alpha_w_momentum = torch.tensor(0)

        self.model_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Initiazed model {model_type} with {self.model_total_params} parameters!")

        record_parents(self, "")

    def forward(self, x, weight=None, alphas=None, feature_indices=None):
        orig_shape = x.shape
        x = x.view(-1, self.n_features)
        if feature_indices is not None:
            for to_delete in range(x.shape[1]):
                if to_delete not in feature_indices:
                    x[:, to_delete] = 0 
        elif self.feature_indices is not None:
            for to_delete in range(x.shape[1]):
                if to_delete not in self.feature_indices:
                    x[:, to_delete] = 0

        # if weight is not None:
        #     old_weights = switch_weights(self, weight)

        if 'AE' in self.model_type:
            return self.model(x.reshape(orig_shape), weight, alphas).reshape(orig_shape)
        else:
            return self.model(x.reshape(orig_shape), weight, alphas)

    def adaptive_weight_decay(self):
        return torch.sum(torch.abs(self.fc1.weight*self.fc1.compute_deg_constants()))
        
    def set_features(self, feature_indices):
        self.feature_indices = set(feature_indices)
        self.model.feature_indices = feature_indices
    
    def alpha_feature_selectors(self):
        return self.model.alpha_feature_selectors()
    
    def feature_normalizers(self):
        return self.model.feature_normalizers()


class RFFRegression(Hypertrainable):
    def __init__(self, d, input_dim, l, num_classes=2, device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
        super().__init__()
        self.embedding = RFFEmbedding(d=d, input_dim=input_dim, l=l, device=device)
        self.fc1 = HyperLinear(d, num_classes)

    def forward(self, x, weight=None, *args, **kwargs):
            
        x = self.embedding(x)
        x = self.fc1(x, weight=weight, **kwargs)
        return x

class RFFRegressionBag(Hypertrainable):
    def __init__(self, d, input_dim, l=1e1, num_classes=2, emb_count=5, device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
        super().__init__()
        ls = np.logspace(1, 15, num=emb_count)
        print(f"Generating {emb_count} RFF embeddings with lengthscales {ls}")
        # ls=[1e5,1e7,1e9,1e11,1e13]
        self.embedding = EmbeddingCombiner(embeddings=[RFFEmbedding(d=d, input_dim=input_dim, renew=False, l=ls[i]) for i in range(emb_count)])
        self.fc1 = HyperLinear(d, num_classes)

    def forward(self, x, weight=None, *args, **kwargs):
            
        x = self.embedding(x)
        x = self.fc1(x, weight=weight, **kwargs)
        return x
        
class LogReg(Hypertrainable, FeatureSelectableTrait):
    def __init__(self, input_dim=28*28, output_dim=10):
        super(LogReg, self).__init__()
        self._input_dim = input_dim
        self.fc1 = HyperLinear(input_dim, output_dim)

    def forward(self, x, weight=None, alphas=None):

        x = x.view(-1, self._input_dim)
        x = self.fc1(x, weight=weight, alphas=alphas)
        return x

    def alpha_feature_selectors(self):
        return torch.abs(self.lin1.weight).data.mean(dim=0)
    
    def feature_normalizers(self):
        return torch.abs(self.lin1.weight).data.mean(dim=0)
    
    def squash(self, x, **kwargs):
        return x

class MLPLarge(Hypertrainable):
    def __init__(self, input_dim=28*28, hidden_dim=1000, output_dim=10, weight_decay=0, dropout_p=0.2, **kwargs):
        super(MLPLarge, self).__init__()
        self._input_dim = input_dim
        self.lin1 = HyperLinear(input_dim, hidden_dim)
        self.lin2 = HyperLinear(hidden_dim, hidden_dim)
        self.lin21 = HyperLinear(hidden_dim, hidden_dim)
        self.lin22 = HyperLinear(hidden_dim, hidden_dim)
        self.lin23 = HyperLinear(hidden_dim, hidden_dim)

        self.lin3 = HyperLinear(hidden_dim, output_dim)

    def forward(self, x, weight=None, alphas=None):

        x = x.view(-1, self._input_dim)

        x = F.relu(self.lin1(x, weight, alphas))
        x = F.relu(self.lin2(x, weight, alphas))
        x = F.relu(self.lin21(x, weight, alphas))
        x = F.relu(self.lin22(x, weight, alphas))
        x = F.relu(self.lin23(x, weight, alphas))

        x = self.lin3(x, weight, alphas)

        return x

class MLP2(Hypertrainable):
    def __init__(self, input_dim=28*28, hidden_dim=1000, output_dim=10, weight_decay=0, dropout_p=0.2, **kwargs):
        super(MLP2, self).__init__()
        self._input_dim = input_dim
        self.lin1 = HyperLinear(input_dim, hidden_dim)
        self.lin2 = HyperLinear(hidden_dim, hidden_dim)
        self.lin3 = HyperLinear(hidden_dim, output_dim)

    def forward(self, x, weight=None, alphas=None):

        x = x.view(-1, self._input_dim)

        x = F.relu(self.lin1(x, weight, alphas))
        x = F.relu(self.lin2(x, weight, alphas))
        x = self.lin3(x, weight, alphas)

        return x

class MLP_sigmoid_relu(Hypertrainable):
    def __init__(self, input_dim=28*28, hidden_dim=1000, output_dim=10, weight_decay=0, dropout_p=0.2, **kwargs):
        super(MLP_sigmoid_relu, self).__init__()
        self._input_dim = input_dim
        self.lin1 = HyperLinear(input_dim, hidden_dim, act="sigmoid_relu")
        self.lin2 = HyperLinear(hidden_dim, hidden_dim, act="sigmoid_relu")
        self.lin3 = HyperLinear(hidden_dim, output_dim)

    def forward(self, x, weight=None, alphas=None):

        x = x.view(-1, self._input_dim)

        x = self.lin1(x, weight, alphas)
        x = self.lin2(x, weight, alphas)
        x = self.lin3(x, weight, alphas)

        return x

class LinearSupernetworkRFF(Hypertrainable):
    def __init__(self, input_dim, output_dim, d=1000):
        super().__init__()
        self.model = Supernetwork(embeddings=[RFFEmbedding(d=d, input_dim=input_dim, renew=False, l=10^i) for i in range(9, 10)], model=torch.nn.Linear(d, output_dim))
        self.input_dim = input_dim

    def forward(self, x, *args, **kwargs):
        x = x.reshape(-1, self.input_dim)
        return self.model(x)

class HyperSequential(torch.nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
    
    def forward(self, input, weight=None, alphas=None):
        for module in self:
            if issubclass(type(module), Hypertrainable):
                input = module(input, weight=weight)
            else:
                input = module(input)
        return input

class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = HyperSequential(
            nn.Dropout(),
            HyperLinear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            HyperLinear(512, 512),
            nn.ReLU(True),
            HyperLinear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, HyperConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x, weight, alphas=None):
        x = self.features(x, weight)
        x = x.view(x.size(0), -1)
        x = self.classifier(x, weight)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = HyperConv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, HyperBatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return HyperSequential(*layers)

vgg_cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}

class MLP(Hypertrainable, FeatureSelectableTrait):
    def __init__(self, input_dim=28*28, hidden_dim=1000, output_dim=10, weight_decay=0, dropout_p=0.2, **kwargs):
        super(MLP, self).__init__()
        self._input_dim = input_dim
        self.feature_selection = FeatureSelection(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, output_dim)
        )
        # self.lin1 = nn.Linear(input_dim, hidden_dim)
        # self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        # self.lin3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, weights=None, alphas=None):

        x = x.view(-1, self._input_dim)
        x = self.feature_selection(x, feature_indices=self.feature_indices)
        x = self.mlp(x)

        return x
    
    def squash(self, *args, **kwargs):
        return self.feature_selection.squash(*args, **kwargs)

    def alpha_feature_selectors(self):
        return self.feature_selection.alphas
    
    def feature_normalizers(self):
        return self.feature_selection.weight.mean(dim=0)

class LinearAE(Hypertrainable, FeatureSelectableTrait, AutoEncoder):
    def __init__(self, input_dim=28*28, dropout_p=0.2, **kwargs):
        super().__init__()
        self._input_dim = input_dim
        # self.input_dropout = nn.Dropout(dropout_p)
        self.feature_selection = FeatureSelection(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128), 
            nn.ReLU(True),
            nn.Dropout(dropout_p))
        self.decoder = nn.Sequential(
            nn.Linear(in_features=128, out_features=input_dim),
        )

    def forward(self, x, *args, **kwargs):
        orig_shape = x.shape
        x = x.view(-1, self._input_dim)
        # x = self.input_dropout(x) #TODO use dropout on the input or not ??
        x = self.feature_selection(x, feature_indices=self.feature_indices)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape(orig_shape)
        return x

    def squash(self, *args, **kwargs):
        return self.feature_selection.squash(*args, **kwargs)

    def alpha_feature_selectors(self):
        return self.feature_selection.alphas
    
    def feature_normalizers(self):
        return self.feature_selection.weight
class AE(Hypertrainable, FeatureSelectableTrait, AutoEncoder):
    def __init__(self, input_dim=28*28, dropout_p=0.1, **kwargs):
        super().__init__()
        self._input_dim = input_dim
        self.dropout_p = dropout_p
        self.input_dropout = nn.Dropout(dropout_p)
        self.feature_selection = FeatureSelection(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128), 
            nn.ReLU(True),
            # nn.Dropout(dropout_p),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(True),
            # nn.Dropout(dropout_p)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(True),
            # nn.Dropout(dropout_p),
            nn.Linear(in_features=128, out_features=input_dim),
        )

    def forward(self, x, *args, **kwargs):
        orig_shape = x.shape
        x = x.view(-1, self._input_dim)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        # x = x*torch.bernoulli(self.squash(self.alpha_feature_selectors())).to(x.device)
        # x = self.input_dropout(x) #TODO use dropout on the input or not ??
        x = self.feature_selection(x, feature_indices=self.feature_indices)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape(orig_shape)
        return x

    def squash(self, *args, **kwargs):
        return self.feature_selection.squash(*args, **kwargs)

    def alpha_feature_selectors(self):
        return self.feature_selection.alphas
    
    def feature_normalizers(self):
        return self.encoder[0].weight.data.mean(dim=0)
