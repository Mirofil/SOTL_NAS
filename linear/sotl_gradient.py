import torch
from utils import hessian
from typing import *
import math
from utils_train import switch_weights, compute_train_loss, calculate_weight_decay

class WeightBuffer:
    def __init__(self, checkpoint_freq, T):
        super().__init__()
        self.weight_buffer = []
        self.checkpoint_freq = checkpoint_freq
        self.T = T

    def add(self, model, intra_batch_idx, clone=True):
        if intra_batch_idx % self.checkpoint_freq == 0:
            if clone is True:
                self.weight_buffer.append([w.clone() for w in model.weight_params()])
            else:
                self.weight_buffer.append(list(model.weight_params()))
        else:
            start = math.floor(intra_batch_idx / self.checkpoint_freq)
            end = min(start + self.checkpoint_freq, self.T - 1)
            self.weight_buffer.append((start, end))
    
    def direct_add(self, weights):
        self.weight_buffer.append(weights)

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

def approx_inverse_hvp(v, f, w, lr, steps = 5):
    # Approximates the inverse-Hessian-vector product v[df/dw]^(-1) by i steps of Neumann series
    # Algorithm is from Optimizing Millions of Hyperparameters by Implicit Differentiation (Lorraine 2020)
    v = v.detach()
    p = v
    for i in range(steps):
        old_v = v
        v = old_v - lr * torch.autograd.grad(f, w, grad_outputs=v, retain_graph=True)[0]
        p += v 
    return lr * p


def sotl_gradient(
    model, criterion, xs, ys, weight_buffer: Sequence, w_lr:float, T:int, 
    grad_outer_loop_order=1,grad_inner_loop_order=1, hvp="exact",
    normalize_a_lr=False, weight_decay_term=0, val_xs = None, val_ys=None, device = 'cuda' if torch.cuda.is_available() else 'cpu',
    mode="joint", inv_hess = "exact", ihvp="exact"
) -> Sequence:

    total_arch_gradient, loss, da, dw, dominant_eigenvalues = None, None, None, None, None
    debug_info = {}
    
    if (grad_inner_loop_order is None) or (grad_inner_loop_order <= 0):
        grad_inner_loop_order = min([len(weight_buffer), len(xs), len(ys)])
    if (grad_outer_loop_order is None) or (grad_outer_loop_order <= 0):
        grad_outer_loop_order = min([len(weight_buffer), len(xs), len(ys)])
        if (val_xs is not None) and (val_ys is not None):
            grad_outer_loop_order = min([grad_outer_loop_order, len(val_xs), len(val_ys)])

    if (
        len(weight_buffer) == 1
    ):  # the other branches won't work because we calculate gradients with weights at t-1 in them
        loss = criterion(model(xs[0], weight_buffer[0], model.fc1.alphas), ys[0])
        da_direct = [y if y is not None else torch.zeros(x.size()) for x,y in zip(model.arch_params(), torch.autograd.grad(loss, model.arch_params(), retain_graph=True, allow_unused=True))]
        total_arch_gradient = da_direct
    else:
        # (1) The outer loop equation is dSoTL/da = sum_{t=T-outer_loop_order)^T dL(w_t, alpha)/da
        # (2) The inner loop equation is dL(w_t, alpha)da = dL(w_t,alpha)/da + dL(w_t,alpha)/dw * -eta sum_{i=t-inner_loop_order}^t d^2L(w_i, alpha)dadw
        
        # OUTER LOOP
        for i in range(
            len(weight_buffer) - 1, max(0, len(weight_buffer)-1-grad_outer_loop_order), -1
        ):

            if (val_xs is not None) and (val_ys is not None):
                top_level_x = val_xs[0]
                top_level_y = val_ys[0]
                
            else:
                top_level_x = xs[i-1]
                top_level_y = ys[i-1]

            top_level_x = top_level_x.to(device)
            top_level_y = top_level_y.to(device)


            # (computing the first two terms in (2)) Gradients using the latest-in-time weights, ie. to compute dL(w_t, alpha)/da, we need dL(w_t,alpha)/dalpha, dL(w_t,alpha)/dw
            top_level_weights = weight_buffer[i]
            old_weights = switch_weights(model, top_level_weights)
            top_level_loss = compute_train_loss(top_level_x, top_level_y, criterion, y_pred=model(top_level_x, top_level_weights), model=model)
            # loss = criterion(model(top_level_x, weight_buffer[i]), top_level_y)
            da_direct = [y if y is not None else torch.zeros(x.size()).to(device) for x,y in zip(model.arch_params(), torch.autograd.grad(top_level_loss, model.arch_params(), retain_graph=True, allow_unused=True))]
            dw = torch.autograd.grad(top_level_loss, top_level_weights, retain_graph=True)

            # no_longer_needed_weights = switch_weights(model, old_weights)

            # (computing the sum of gradients in (1))
            for j in range(i-1, max(-1, i - grad_inner_loop_order-1), -1):

                x = xs[j].to(device)
                y = ys[j].to(device)

                loss2 = compute_train_loss(x, y, criterion, y_pred=model(x, weight_buffer[j]), model=model)

                if inv_hess == "ift":
                    inv_hess_matrices_dwdw = [torch.inverse(hessian(
                        loss2 * 1, weight_buffer[j][idx], weight_buffer[j][idx]
                    )) for idx in range(len(weight_buffer[j]))]
                elif inv_hess == "exact":
                    prods = [torch.eye(w.shape[1]) for w in weight_buffer[j]]
                    for k in range(min(i-2, j), j, -1):
                        loss3 = compute_train_loss(x=xs[k].to(device), y=ys[k].to(device), criterion=criterion, 
                            y_pred=model(xs[k].to(device), weight_buffer[k]), model=model)
                        hess_matrices_dwdw = [hessian(loss3*1, w, w) for w in weight_buffer[k]]

                        for idx, (prod, hess) in enumerate(zip(prods, hess_matrices_dwdw)):
                            prods[idx] = prods[idx] @ (torch.eye(hess.shape[0]) - hess)
                    
                    inv_hess_matrices_dwdw = prods


                elif inv_hess == "id":
                    inv_hess_matrices_dwdw = [torch.eye(weight_buffer[j][idx].shape[0]) for idx in range(len(weight_buffer[j]))] # TODO THERE SHOULD BE A RANGE TO ACCOMMODATE ALL TIMESTEPS
                    

                ihvp_vecs = [0 for _ in range(len(dw))]
                for idx, (grad_w, inverse_hess_dwdw) in enumerate(zip(dw, inv_hess_matrices_dwdw)):

                    if inv_hess != "id":
                        if ihvp == "ift":
                            ihvp_vec = torch.matmul(grad_w, inverse_hess_dwdw)
                        elif ihvp == "exact":
                            ihvp_vec = torch.matmul(grad_w, inverse_hess_dwdw)
                        elif ihvp == "neumann":
                            dL_train_dw = torch.autograd.grad(loss2, model.weight_params(), create_graph=True)
                            ihvp_vec = approx_inverse_hvp(v=grad_w, f=dL_train_dw, w=list(model.weight_params()), lr=w_lr, steps=500)
                        ihvp_vecs[idx] = ihvp_vec
                    else:
                        ihvp_vecs[idx] = grad_w

                if hvp == "exact":
                    # INNER LOOP - computation of gradients within sum

                    # NOTE this exact pathway only makes sense for the linear model because we materialize the inverse Hessian. So playing with indexes for multiple arch/weight Hessian pairs here is not very meaningful either

                    hessian_matrices_dadw = [hessian(
                        loss2 * 1, weight_buffer[j][idx], arch_param
                    ) for arch_param in model.arch_params() for idx in range(len(weight_buffer[j]))]


                    # if hasattr(model, "fc1"):
                    #     loss2 = compute_train_loss(x, y, criterion, y_pred=model(x, weight_buffer[j]), model=model)
                    #     arch_hessian = hessian(loss2*1, model.fc1.alphas, model.fc1.alphas) # We are interested in the max_deg/sigmoid parameters.. how to make the arch_params handling more general here?
                    #     eigenvalues = torch.symeig(arch_hessian)
                    #     dominant_eigenvalues = eigenvalues.eigenvalues[-1]# Eigenvalues are returned in ascending order!

                    second_order_terms = []
                    for hess_dawdw in hessian_matrices_dadw:
                        for ihvp_vec, inverse_hess_dwdw in zip(ihvp_vecs, inv_hess_matrices_dwdw):
                            jvp = torch.matmul(ihvp_vec, hess_dawdw)
                            second_order_terms.append(jvp)

                elif hvp == "finite_diff":
                    # INNER LOOP
                    # DARTS footnotes suggest to divide by L2 norm of the gradient
                    norm = torch.cat([w.view(-1) for w in dw]).norm()
                    eps = 0.0001 / norm


                    # w+ = w_{t-1} + eps*dL(w_t,alpha)dw
                    with torch.no_grad():
                        for p, d in zip(weight_buffer[j], dw):
                            p.add_(eps * d)

                    old_weights = switch_weights(model, weight_buffer[j])
                    loss_pos = compute_train_loss(x, y, criterion, y_pred=model(x, weight_buffer[j]), model=model)

                    
                    dalpha_pos = [a if (a is not None) else torch.zeros(list(model.arch_params())[i].size()).to(device) for i, a in enumerate(torch.autograd.grad(
                        loss_pos, model.arch_params(), allow_unused=True
                    ))]  # dalpha { L_trn(w+) }
                    no_longer_needed_weights = switch_weights(model, old_weights)

                    # w- = w_{t-1} - eps*dL(w_t,alpha)dw
                    with torch.no_grad():
                        for p, d in zip(weight_buffer[j], dw):
                            p.subtract_(2.0 * eps * d)

                    old_weights = switch_weights(model, weight_buffer[j])
                    
                    loss_neg = compute_train_loss(x, y, criterion, y_pred=model(x, weight_buffer[j]), model=model)
                    dalpha_neg = [a if a is not None else torch.zeros(list(model.arch_params())[i].size()).to(device) for i, a in enumerate(torch.autograd.grad(
                        loss_neg, model.arch_params(), allow_unused=True
                    ))]  # dalpha { L_trn(w-) }
                    no_longer_needed_weights = switch_weights(model, old_weights)

                    # recover w
                    with torch.no_grad():
                        for p, d in zip(weight_buffer[j], dw):
                            p.add_(eps * d)

                    second_order_terms = [
                        (p - n) / (2.0 * eps) for p, n in zip(dalpha_pos, dalpha_neg)
                    ]
                else:
                    raise NotImplementedError

                total_arch_gradient_local = [
                    -w_lr*da1 for da1 in second_order_terms
                ]
                if total_arch_gradient is None:
                    total_arch_gradient = total_arch_gradient_local

                else:
                    for g1, g2 in zip(total_arch_gradient, total_arch_gradient_local):
                        g1.add_(g2)

            
            for arch_grad, direct_grad in zip(total_arch_gradient, da_direct):
                arch_grad.add_(direct_grad)


    if normalize_a_lr:
        for g in total_arch_gradient:
            g.multiply_(T/grad_inner_loop_order)

    return {"total_arch_gradient":total_arch_gradient, 
    "dominant_eigenvalues":dominant_eigenvalues, 
    "da_direct":da_direct,
    "dw_direct":dw}

