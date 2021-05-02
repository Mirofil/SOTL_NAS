import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sotl_nas.datasets.rff_mnist import *
from sotl_nas.models.base import Hypertrainable
from sotl_nas.models.traits import FeatureSelectableTrait
from torch import Tensor


class LinearSquash(torch.nn.Linear, Hypertrainable, FeatureSelectableTrait):
    def __init__(self, in_features, out_features, bias, squash_type="softmax", **kwargs) -> None:
        super().__init__(in_features,out_features,bias)
        self.alphas = torch.nn.Parameter(torch.zeros(1, in_features))
        self.squash_type = squash_type


    def forward(self, input: Tensor, weight: Tensor = None, alphas: Tensor = None) -> Tensor:
        if weight is None:
            weight = self.weight
            return F.linear(input, weight*self.squash(alphas), self.bias)
        else:
            extracted_params = {k:weight[self.parent_path+"."+k] for k, v in self.named_weight_params()}
            extracted_params["weight"] = extracted_params["weight"]*self.squash(alphas)
            return F.linear(input, **extracted_params)
    
    def squash_constants(self):
        return self.squash(self.alphas)
    
    def squash(self, *args, **kwargs):
        if self.squash_type == "softmax":
            return F.softmax(*args, **kwargs)
        elif self.squash_type == "sigmoid":
            return torch.sigmoid(*args, **kwargs)

    def alpha_feature_selectors(self):
        return self.alphas
    
    def feature_normalizers(self):
        return self.weight.mean(dim=0)

class FeatureSelection(Hypertrainable):
    def __init__(self, in_features, squash_type="sigmoid", **kwargs) -> None:
        super().__init__()
        # self.weight = torch.ones((1, in_features))
        self.register_buffer('weight', torch.ones((1, in_features)))
        self.feature_indices = {i for i in range(in_features)}

        self.alphas = torch.nn.Parameter(torch.ones(1, in_features))
        self.squash_type = squash_type

    def forward(self, x: Tensor, feature_indices=None) -> Tensor:
        if feature_indices is not None:
            for to_delete in range(x.shape[1]):
                if to_delete not in feature_indices:
                    x[:, to_delete] = 0 
        elif self.feature_indices is not None:
            for to_delete in range(x.shape[1]):
                if to_delete not in self.feature_indices:
                    x[:, to_delete] = 0 
        return x * self.squash(self.alphas)

    def squash(self, *args, **kwargs):
        if self.squash_type == "softmax":
            return F.softmax(*args, **kwargs)
        elif self.squash_type == "sigmoid":
            return torch.sigmoid(*args, **kwargs)

class Supernetwork(Hypertrainable):
    def __init__(self, embeddings, model):
        super().__init__()
        self.emb_layer = EmbeddingCombiner(embeddings, trainable=False, mode="random") 
        self.model = model
    
    def forward(self, x, **kwargs):
        x = self.emb_layer(x)
        x = self.model(x)

        return x

class EmbeddingCombiner(Hypertrainable):
    def __init__(self, embeddings, device='cuda' if torch.cuda.is_available() else 'cpu', trainable=True, mode ="softmax", **kwargs) -> None:
        super().__init__()
        # NOTE embeddings should be in a Python list (NOT ParameterList) so that they are not seen as Parameters by NN.module! The parameters of each embedding should not be optimized!
        self.embeddings = [emb.to(device) for emb in embeddings]
        if mode == "softmax":
            self.alpha_lin_comb = torch.nn.Parameter(torch.tensor([0 for _ in range(len(embeddings))], dtype=torch.float32), requires_grad = trainable)
        self.softmax = torch.nn.Softmax(dim=0)
        self.device=device
        self.mode = mode

    def forward(self, x):
        if self.mode == "softmax":
            weights = self.softmax(self.alpha_lin_comb)
        elif self.mode == "random":
            weights = [0 for i in range(len(self.embeddings))]
            picked_embedding = random.randint(0, len(self.embeddings)-1)
            weights[picked_embedding] = 1
            weights = torch.tensor(weights)
        embs = [emb(x) if w != 0 else torch.zeros_like(emb.d) for emb,w in zip(self.embeddings, weights)]

        # print(embs)
        # embs = torch.sum([emb*w for emb, w in zip(embs, weights)], dim=1)
        # for emb in embs:
        #     print(emb.shape)
        embs = sum([emb*w for emb, w in zip(embs, weights)])
        return embs

class RFFEmbedding(Hypertrainable):
    def __init__(self, d, input_dim, l, renew=False, trainable=True, device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs) -> None:
        super().__init__()
        # self.embedding = build_embedding(d=d, k=input_dim, l=l)
        # self.w = torch.rand(d, input_dim)
        self.d = torch.tensor(d)
        self.input_dim = input_dim
        self.device =device


        self.w = torch.normal(0,1,(self.d, self.input_dim)).reshape(self.d, self.input_dim).to(self.device)
        self.b = torch.tensor(2*np.pi * np.random.rand(d)).to(self.device)
        self.alpha_l = torch.nn.Parameter(torch.tensor(l, dtype=torch.float32), requires_grad=trainable)
        self.renew = renew
        self.counter = 0
    def forward(self, x):
        return self.embedding(x)

    def embedding(self, X):
        if self.renew and self.counter % 1 == 0:
            self.w = torch.normal(0,1,(self.d, self.input_dim)).reshape(self.d, self.input_dim).to(self.device)
            self.b = torch.tensor(2*np.pi * np.random.rand(self.d)).to(self.device)
        self.counter += 1
        n = X.shape[0]
        #TODO should there be sqrt near alpha_l?
        fs = ((self.w/self.alpha_l) @ X.T).T + torch.tensor(torch.repeat_interleave(self.b, 1, axis=0)).to(self.device)
        return torch.sqrt(2/self.d) * torch.cos(fs).float()


class LinearMaxDeg(torch.nn.Linear, Hypertrainable):
    def __init__(self, in_features, out_features, bias=True, degree=30, device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs) -> None:
        super().__init__(in_features, out_features, bias)
        self.alphas = torch.nn.Parameter(torch.tensor([degree], dtype=torch.float32).unsqueeze(dim=0))
        constants = [0]
        for i in range(1,self.in_features+1):
            constants.append(-i*1) # the multiplicative constant here depends on the width of the soft step function used
            constants.append(-i*1)
        constants = constants[:self.in_features]
        self.degree = self.alphas
        self.alpha_constants = torch.nn.Parameter(torch.tensor(constants,dtype=torch.float32).unsqueeze(dim=0), requires_grad=False)
        self.device=device

    def forward(self, input: Tensor, weight: Tensor = None, alphas: Tensor = None) -> Tensor:
        if weight is None:
            weight = self.weight
            return F.linear(input, weight.to(self.device)*self.compute_deg_constants(alphas=alphas).to(self.alpha_constants.device), self.bias)
        else:
            extracted_params = {k:weight[self.parent_path+"."+k] for k, v in self.named_weight_params()}
            extracted_params["weight"] = extracted_params["weight"].to(self.device)*self.compute_deg_constants(alphas=alphas).to(self.alpha_constants.device)
            return F.linear(input, **extracted_params)


    def compute_deg_constants(self, alphas = None):
        if alphas is None:
            alphas = self.degree
        return self.squished_tanh(alphas+self.alpha_constants)

    @staticmethod
    def squished_tanh(x, plot=False):
        if plot:
            xs = np.linspace(-5,5,100)
            ys = [(F.tanh(1*torch.tensor(elem))+1)/2 for elem in xs]
            plt.plot(xs,ys)

        return (F.tanh(x)+torch.tensor(1))/2
    
    def squash(self, *args, **kwargs):
        return self.squished_tanh(*args, **kwargs)



class HyperLinear(torch.nn.Linear, Hypertrainable):
    def __init__(self, in_features, out_features, bias=True, act=None, **kwargs):
        super().__init__(in_features, out_features, bias=bias)
        if act is None or act == "id":
            self.act = self.id
        elif act == "sigmoid_relu":
            self.act = self.act_sigmoid_relu
            self.alpha_sigmoid_relu_param = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

    def forward(self, input: Tensor, weight: Tensor = None, alphas: Tensor = None, **kwargs) -> Tensor:
        if weight is None:
            return self.act(F.linear(input, self.weight, bias=self.bias))
        else:
            extracted_params = {k:weight[self.parent_path+"."+k] for k, v in self.named_weight_params()}
            return self.act(F.linear(input, weight=extracted_params["weight"], bias=extracted_params["bias"]))

    def act_sigmoid_relu(self, x):
        # Interpolates between ReLU and sigmoid
        a = torch.sigmoid(self.alpha_sigmoid_relu_param) # need to squish the a parameter into [0, 1]
        numerator = 2*a+(1-a)*(x+torch.sqrt(torch.pow(x, 2) + torch.pow(a, 2)))
        denominator = 2 + 2*a*torch.exp(-x)
        return numerator/denominator

    @staticmethod
    def id(x):
        return x


class HyperConv2d(torch.nn.Conv2d, Hypertrainable):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, input: Tensor, weight: Tensor = None, alphas: Tensor = None, **kwargs) -> Tensor:
        if weight is None:
            return super().forward(input)
        else:
            extracted_params = {k:weight[self.parent_path+"."+k] for k, v in self.named_weight_params()}
            return F.conv2d(input, padding=self.padding, stride=self.stride, **extracted_params)
class HyperBatchNorm2d(torch.nn.BatchNorm2d, Hypertrainable):
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor, weight: Tensor = None, alphas: Tensor = None, **kwargs) -> Tensor:
        if weight is None:
            return super().forward(input)
        else:
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum
            if self.training:
                bn_training = True
            else:
                bn_training = (self.running_mean is None) and (self.running_var is None)
            extracted_params = {k:weight[self.parent_path+"."+k] for k, v in self.named_weight_params()}
            return F.batch_norm(
                        input,
                        # If buffers are not to be tracked, ensure that they won't be updated
                        self.running_mean if not self.training or self.track_running_stats else None,
                        self.running_var if not self.training or self.track_running_stats else None,
                        training=bn_training, momentum=exponential_average_factor, eps=self.eps, **extracted_params)
