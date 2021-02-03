import torch
from utils import hessian
from typing import *
import math

class WeightBuffer:
    def __init__(self, checkpoint_freq, T):
        super().__init__()
        self.weight_buffer = []
        self.checkpoint_freq = checkpoint_freq
        self.T = T

    def add(self, model, intra_batch_idx):
        if intra_batch_idx % self.checkpoint_freq == 0:
            self.weight_buffer.append([w.clone() for w in model.weight_params()])
        else:
            start = math.floor(intra_batch_idx / self.checkpoint_freq)
            end = min(start + self.checkpoint_freq, self.T - 1)
            self.weight_buffer.append((start, end))

    def __len__(self):
        return len(self.weight_buffer)

    def __getitem__(self, key):
        return self.get(key)

    def get(self, i: int):
        if not isinstance(self.weight_buffer[i][0], (int)):
            return self.weight_buffer[i]
        else:
            start_w = self.weight_buffer[i][0]
            end_w = self.weight_buffer[i][1]
            return [start + (end - start) / 2 for (start, end) in zip(start_w, end_w)]

    def clear(self):
        self.weight_buffer = []

def sotl_gradient(
    model, criterion, xs, ys, weight_buffer: Sequence, w_lr:float, T:int, outer_loop_order=1,inner_loop_order=1, hvp="exact", normalize_a_lr=True, weight_decay_term=0, val_xs = None, val_ys=None
) -> Sequence:
    total_arch_gradient = None
    loss = None
    da=None
    dw=None
    

    if (inner_loop_order is None) or (inner_loop_order <= 0):
        inner_loop_order = min([len(weight_buffer), len(xs), len(ys)])
    if (outer_loop_order is None) or (outer_loop_order <= 0):
        outer_loop_order = min([len(weight_buffer), len(xs), len(ys)])
        if (val_xs is not None) and (val_ys is not None):
            outer_loop_order = min([outer_loop_order, len(val_xs), len(val_ys)])

    if (
        len(weight_buffer) == 1
    ):  # the other branches won't work because we calculate gradients with weights at t-1 in them
        loss = criterion(model(xs[0], weight_buffer[0], model.fc1.alphas), ys[0])
        da = [y if y is not None else torch.zeros(x.size()) for x,y in zip(model.arch_params(), torch.autograd.grad(loss, model.arch_params(), retain_graph=True, allow_unused=True))]
        total_arch_gradient = da
    else:
        # The outer loop equation is dSoTL/da = sum_{t=T-outer_loop_order)^T dL(w_t, alpha)/da
        # The inner loop equation is dL(w_t, alpha)da = dL(w_t,alpha)/da + dL(w_t,alpha)/dw * -eta sum_{i=t-inner_loop_order}^t d^2L(w_i, alpha)dadw
        
        # OUTER LOOP
        for i in range(
            len(weight_buffer) - 1, max(0, len(weight_buffer)-1-outer_loop_order), -1
        ):
            if (val_xs is not None) and (val_ys is not None):
                top_level_x = val_xs[0]
                top_level_y = val_ys[0]
                
            else:
                top_level_x = xs[i]
                top_level_y = ys[i]


            loss = criterion(model(top_level_x, weight_buffer[i], model.fc1.alphas), top_level_y)
            da = [y if y is not None else torch.zeros(x.size()) for x,y in zip(model.arch_params(), torch.autograd.grad(loss, model.arch_params(), retain_graph=True, allow_unused=True))]
            dw = torch.autograd.grad(loss, weight_buffer[i], retain_graph=True)
            if hvp == "exact":
                # INNER LOOP
                for j in range(i, max(0, i - inner_loop_order), -1):
                    param_norm = 0
                    if model.alpha_weight_decay > 0:
                        for weight in weight_buffer[j - 1]:
                            param_norm = param_norm + torch.pow(weight.norm(2), 2)
                    loss2 = criterion(
                        model(xs[i], weight_buffer[j - 1], model.fc1.alphas), ys[i]
                    ) + param_norm*model.alpha_weight_decay
                    hessian_matrices = [hessian(
                        loss2 * 1, weight_buffer[j - 1], arch_param
                    ).reshape(
                        model.fc1.weight.size()[1], arch_param.size()[1]
                    ) for arch_param in model.arch_params()]
                    # hessian_matrix = hessian(
                    #     loss2 * 1, weight_buffer[j - 1][0], model.fc1.alphas
                    # ).reshape(
                    #     model.fc1.weight.size()[1], model.fc1.alphas.size()[1]
                    # )  # TODO this whole line is WEIRD. It should be made more general to allow multiple alphas ass well

                    second_order_terms = [torch.matmul(dw[0], hessian_matrix) for hessian_matrix in hessian_matrices]

                    if total_arch_gradient is None:
                        total_arch_gradient = [0 for _ in range(len(da))]
                    for k, a_grad in enumerate(da):

                        total_arch_gradient[k] += a_grad + (
                            -w_lr * second_order_terms[k]
                        ) 

            elif hvp == "finite_diff":
                # INNER LOOP
                for j in range(i, max(0, i - inner_loop_order), -1):
                    # DARTS footnotes suggest to divide by L2 norm of the gradient
                    norm = torch.cat([w.view(-1) for w in dw]).norm()
                    eps = 0.0001 / norm

                    # w+ = w_{t-1} + eps*dL(w_t,alpha)dw
                    with torch.no_grad():
                        for p, d in zip(weight_buffer[j - 1], dw):
                            p.add_(eps * d)
                    param_norm = 0
                    if model.alpha_weight_decay > 0:
                        for weight in weight_buffer[j - 1]:
                            param_norm = param_norm + torch.pow(weight.norm(2), 2)
                    loss2 = criterion(
                        model(xs[j], weight_buffer[j - 1], model.fc1.alphas), ys[j]
                    ) + param_norm*model.alpha_weight_decay
                    dalpha_pos = [x for x in torch.autograd.grad(
                        loss2, model.arch_params(), allow_unused=True
                    ) if x is not None]  # dalpha { L_trn(w+) }

                    # w- = w_{t-1} - eps*dL(w_t,alpha)dw
                    with torch.no_grad():
                        for p, d in zip(weight_buffer[j - 1], dw):
                            p.subtract_(2.0 * eps * d)

                    param_norm = 0
                    if model.alpha_weight_decay > 0:
                        for weight in weight_buffer[j - 1]:
                            param_norm = param_norm + torch.pow(weight.norm(2), 2)
                    loss3 = criterion(
                        model(xs[j], weight_buffer[j - 1], model.fc1.alphas), ys[j]
                    ) + param_norm*model.alpha_weight_decay
                    dalpha_neg = [x for x in torch.autograd.grad(
                        loss3, model.arch_params(), allow_unused=True
                    ) if x is not None]  # dalpha { L_trn(w-) }

                    # recover w
                    with torch.no_grad():
                        for p, d in zip(weight_buffer[j - 1], dw):
                            p.add_(eps * d)

                    second_order_term = [
                        -w_lr * (p - n) / (2.0 * eps) for p, n in zip(dalpha_pos, dalpha_neg)
                    ]
                    total_arch_gradient_local = [
                        da1 + da2 for (da1, da2) in zip(second_order_term, da)
                    ]
                    if total_arch_gradient is None:
                        total_arch_gradient = total_arch_gradient_local

                    else:
                        for g1, g2 in zip(total_arch_gradient, total_arch_gradient_local):
                            g1.add_(g2)
            else:
                raise NotImplementedError

    if normalize_a_lr:
        for g in total_arch_gradient:
            g.multiply_(T/inner_loop_order)
    return total_arch_gradient

