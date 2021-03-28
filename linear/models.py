import torch
from layers import LinearSquash, LinearMaxDeg, FlexibleLinear, FeatureSelection, RFFEmbedding, EmbeddingCombiner
import torch.nn as nn
import torch.nn.functional as F
import itertools
from traits import FeatureSelectableTrait, AutoEncoder, Regressor, Classifier
import numpy as np
from utils_train import switch_weights, record_parents
from models_base import Hypertrainable

class SoTLNet(Hypertrainable):
    def __init__(self, num_features = 2, model_type = "softmax_mult", task="whatever",
     extra_weight_decay=0, n_classes=1, cfg=None, alpha_lr=None, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.num_features = num_features
        self.n_classes = n_classes
        self.cfg = cfg

    

        if model_type == "softmax_mult":
            self.fc1 = LinearSquash(num_features, n_classes, bias=False, squash_type="softmax", **kwargs)
            self.model = self.fc1
        elif model_type == "sigmoid":
            self.fc1 = LinearSquash(num_features, n_classes, bias=False, squash_type="sigmoid", **kwargs)

            self.model = self.fc1
        elif model_type == "rff":
            l = cfg["l"] if "l" in cfg.keys() else 1
            self.model = RFFRegression(1000, num_features, l, **kwargs)
        elif model_type == "rff_bag":
            self.model = RFFRegressionBag(1000, num_features, **kwargs)
        elif model_type == "max_deg":
            self.fc1 = LinearMaxDeg(num_features, n_classes, bias=False, **kwargs)

            self.model = self.fc1
        elif model_type == "linear":
            self.fc1 = FlexibleLinear(num_features, n_classes, bias=False)
            self.model = self.fc1
        elif model_type == "MNIST" or model_type == "MLP":
            self.model = MLP(input_dim=num_features,hidden_dim=1000,output_dim=n_classes, weight_decay=extra_weight_decay)
        elif model_type == "pt_logistic_l1":
            self.model = LogReg(input_dim=num_features, output_dim=n_classes)
            print("Setting (overriding?) default decay values for logistic L1")
            self.config["w_decay_order"] = 1
            self.config["w_weight_decay"] = 1
            self.config["a_decay_order"] = 0
        elif model_type == "log_reg":
            self.model = LogReg(input_dim=num_features, output_dim=n_classes)
        elif model_type == "AE":
            self.model = AE(input_dim=num_features)
        elif model_type == "linearAE":
            self.model = LinearAE(input_dim=num_features)
        else:
            raise NotImplementedError
        self.alphas = []
        self.feature_indices = None
        self.model.feature_indices = self.feature_indices
        if extra_weight_decay > 0:
            self.alpha_weight_decay = torch.nn.Parameter(torch.tensor([extra_weight_decay], dtype=torch.float32, requires_grad=True).unsqueeze(dim=0))
        else:
            self.alpha_weight_decay = torch.tensor(0)
        if alpha_lr is not None:
            self.alpha_lr = torch.nn.Parameter(torch.tensor(alpha_lr, dtype=torch.float32, requires_grad=True).unsqueeze(dim=0))
        else:
            self.alpha_lr = torch.tensor(0)

        record_parents(self, "")

    def forward(self, x, weight=None, alphas=None, feature_indices=None):
        orig_shape = x.shape
        x = x.view(-1, self.num_features)
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
            return self.model(x, weight, alphas).reshape(orig_shape)
        else:
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


class RFFRegression(Hypertrainable):
    def __init__(self, d, input_dim, l, num_classes=2, device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
        super().__init__()
        self.embedding = RFFEmbedding(d=d, input_dim=input_dim, l=l, device=device)
        self.fc1 = FlexibleLinear(d, num_classes)

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
        self.fc1 = FlexibleLinear(d, num_classes)


    def forward(self, x, weight=None, *args, **kwargs):
            
        x = self.embedding(x)
        x = self.fc1(x, weight=weight, **kwargs)
        return x
        
class LogReg(Hypertrainable, FeatureSelectableTrait):
    def __init__(self, input_dim=28*28, output_dim=10):
        super(LogReg, self).__init__()
        self._input_dim = input_dim
        self.fc1 = FlexibleLinear(input_dim, output_dim)

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