import torch
from utils import hessian
from typing import *

def sotl_gradient(model, criterion, xs, ys, weight_buffer, w_lr, order=1, hvp = "exact") -> Sequence:
    total_arch_gradient = None
    if order is None or order <= 0:
        order = len(weight_buffer)
    if hvp == "exact":
        for i in range(len(weight_buffer)-1, max(0, len(weight_buffer)-1-order), -1):
            loss = criterion(model(xs[i], weight_buffer[i][0], model.fc1.alphas), ys[i])
            da = torch.autograd.grad(loss, model.arch_params(), retain_graph=True)[0]
            dw = torch.autograd.grad(loss, weight_buffer[i][0], retain_graph=True)[0]

            loss2 = criterion(model(xs[i], weight_buffer[i-1][0], model.fc1.alphas), ys[i])
            hessian_matrix = hessian(loss2*1, weight_buffer[i-1][0], model.fc1.alphas).reshape(model.fc1.weight.size()[1],model.fc1.alphas.size()[1]) # TODO this whole line is WEIRD

            second_order_term = torch.matmul(dw, hessian_matrix)

            if total_arch_gradient is None:
                total_arch_gradient = [0]
            total_arch_gradient[0] += da + (-w_lr*second_order_term) # TODO this does not work if there are multiple arch parameter tensors! See the handling in finite diff code below

    elif hvp == "finite_diff":
        for i in range(len(weight_buffer)-1, max(0, len(weight_buffer)-1-order), -1):
            loss = criterion(model(xs[i], weight_buffer[i][0], model.fc1.alphas), ys[i])
            da = torch.autograd.grad(loss, model.arch_params(), retain_graph=True)
            dw = torch.autograd.grad(loss, weight_buffer[i][0], retain_graph=True)[0]

            # Footnotes suggest to divide by L2 norm of the gradient
            norm = torch.cat([w.view(-1) for w in dw]).norm()
            eps = 0.0001 / norm

            # w+ = w + eps*dw`
            with torch.no_grad():
                for p, d in zip(weight_buffer[i-1][0], dw):
                    p.add_(eps * d)

            loss2 = criterion(model(xs[i], weight_buffer[i-1][0], model.fc1.alphas), ys[i])
            dalpha_pos = torch.autograd.grad(
                loss2, model.arch_params()
            )  # dalpha { L_trn(w+) }

            # w- = w - eps*dw`
            with torch.no_grad():
                for p, d in zip(weight_buffer[i-1][0], dw):
                    p.subtract_(2.0 * eps * d)
            loss3 = criterion(model(xs[i], weight_buffer[i-1][0], model.fc1.alphas), ys[i])
            dalpha_neg = torch.autograd.grad(
                loss3, model.arch_params()
            )  # dalpha { L_trn(w-) }

            # recover w
            with torch.no_grad():
                for p, d in zip(weight_buffer[i-1][0], dw):
                    p.add_(eps * d)

            second_order_term = [-w_lr*(p - n) / (2.0 * eps) for p, n in zip(dalpha_pos, dalpha_neg)]
            total_arch_gradient_local = [da1 + da2 for (da1, da2) in zip(second_order_term, da)]
            if total_arch_gradient is None:
                total_arch_gradient = total_arch_gradient_local
            else:
                for g1, g2 in zip(total_arch_gradient, total_arch_gradient_local):
                    g1.add_(g2)


    return total_arch_gradient
