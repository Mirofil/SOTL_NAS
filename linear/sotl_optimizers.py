import torch
from torch import Tensor
from torch.optim import SGD
from typing import *
class HyperSGD(SGD):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, grad=False):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.grad = grad

    def step(self, closure=None, grad=None):
        if grad is None:
            grad= self.grad
        with torch.set_grad_enabled(grad):
            return super().step(closure)

    @staticmethod
    def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool):
        r"""Functional API that performs SGD algorithm computation.
        See :class:`~torch.optim.SGD` for details.
        """

        for i, param in enumerate(params):

            d_p = d_p_list[i]
            if weight_decay != 0:
                d_p = d_p.add(param, alpha=weight_decay)

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

            param = param.detach() - lr * d_p
            # param.add_(d_p, alpha=-lr)
