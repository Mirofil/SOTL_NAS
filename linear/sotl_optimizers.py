import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import SGD
from typing import *


class HyperSGD(SGD):
    def __init__(self, params, named_params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, grad=False, T=1):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.T = T+1 # T is the length of unrollment; used for modular arithmetic in computing the histories here
        self.i = 1 # the current position in the circular buffer of momentum histories
        self.grad = grad

        for group in self.param_groups:
            group["momentum_buffer"] = [{k:None for k, v in named_params.items()} for _ in range(T+1)] # First position is used as buffer from last rollout

    # def step(self, closure=None, grad=None):
    #     if grad is None:
    #         grad= self.grad
    #     with torch.set_grad_enabled(grad):
    #         return super().step(closure)

    def step(self, grads, config, weight_buffer):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            momentum_buffer = group["momentum_buffer"]

            new_weights, new_momentum = self.sgd(grads,
                  weight_decay,
                  momentum,
                  lr,
                  dampening,
                  nesterov,
                  config,
                  weight_buffer,
                  momentum_buffer)

            # update momentum_buffers in state
            if momentum > 0:
                next_index = (self.i+1) % self.T
                cur_index = self.i % self.T
                momentum_buffer[cur_index] = new_momentum

                if next_index == 0: # Remember self.T is length of period + 1 since the first pos is used as init. Hence if this If is true, then self.i=last position in buffer
                    # Uses the last momentum of the rollout as initialization for the next rollout buffer
                    momentum_buffer[next_index] = {k:v.detach() for k, v in momentum_buffer[cur_index].items()}
        self.i += 1

        return new_weights

    def sgd(self, grads,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        config,
        weight_buffer,
        momentum_buffer
        ):
        r"""Functional API that performs SGD algorithm computation.
        See :class:`~torch.optim.SGD` for details.
        """
        new_weights = {}
        new_momentum = {}
        # for i, param in enumerate(params):

        #     d_p = d_p_list[i]
        #     if weight_decay != 0:
        #         d_p = d_p.add(param, alpha=weight_decay)

        #     if momentum != 0:
        #         buf = momentum_buffer_list[i]

        #         if buf is None:
        #             new_buf = torch.clone(d_p).detach()
        #             momentum_buffer_list[i] = new_buf
        #         else:
        #             new_buf = new_buf * momentum + d_p * (1 - dampening)
        #             # buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

        #         if nesterov:
        #             d_p = d_p.add(new_buf, alpha=momentum)
        #         else:
        #             d_p = buf

        #     param.add_(d_p, alpha=-lr)

        for (w_name, w), dw in zip(weight_buffer[-1].items(), grads):
            if type(config["w_lr"]) is float or not config["softplus_alpha_lr"]:
                if momentum == 0:
                    new_weights[w_name] = w - config["w_lr"]*dw # Manual SGD update that creates new nodes in the computational graph
                    new_momentum[w_name] = None
                else:
                    # equation from https://stats.stackexchange.com/questions/179915/whats-the-difference-between-momentum-based-gradient-descent-and-nesterovs-acc
                    if momentum_buffer[self.i-1][w_name] is not None:
                        momentum_update = momentum * (momentum * momentum_buffer[self.i-1][w_name] - config["w_lr"]*dw) - config["w_lr"]*dw
                    else:
                        momentum_update = -config["w_lr"]*dw
                    new_weights[w_name] = w + momentum_update
                    new_momentum[w_name] = momentum_update

            else: # TODO finish
                new_weights[w_name] = w - F.softplus(config["w_lr"], config["softplus_beta"])*dw # Manual SGD update that creates new nodes in the computational graph

        return new_weights, new_momentum
        
