import torch

class Hypertrainable(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Hypertrainable, self).__init__()

    def forward(self, x):
        raise NotImplementedError

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

    def named_arch_params(self):
        for n,p in self.named_parameters():
            if 'alpha' in n and p.requires_grad:
                yield (n, p)
            else:
                continue