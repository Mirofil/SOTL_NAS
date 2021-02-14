import torch
import torch.nn as nn
from torch.optim import SGD, Adam

def switch_weights(model, weight_buffer_elem):
    with torch.no_grad():
        old_weights = [w.clone() for w in model.weight_params()]

        for w_old, w_new in zip(model.weight_params(), weight_buffer_elem):
            w_old.copy_(w_new)
    
    return old_weights

def hinge_loss(x,y, threshold):
    if abs(x-y) <= threshold:
        return 0
    elif x > y:
        return x -y - threshold
    elif x < y:
        return y - x + threshold


def get_optimizers(model, config):
    w_optimizer = SGD(model.weight_params(), lr=config["w_lr"], momentum=config["w_momentum"], weight_decay=config["w_weight_decay"])
    
    if config["train_arch"]:
        a_optimizer = SGD(model.arch_params(), lr=config["a_lr"], momentum=config["a_momentum"], weight_decay=config["a_weight_decay"])
    else:
        # Placeholder optimizer that won't do anything - but the parameter list cannot be empty
        a_optimizer = None

    return w_optimizer, a_optimizer


def get_criterion(model_type):
    criterion=None
    if model_type in ["MNIST", "log_regression"]:
        criterion = torch.nn.CrossEntropyLoss()
    elif model_type in ["max_deg", "softmax_mult", "linear", "fourier", "polynomial"]:
        criterion = torch.nn.MSELoss()
    
    return criterion

def train_normal(
    num_epochs, model, dset_train, batch_size, grad_clip, logging_freq, optim="sgd", **kwargs
):
    train_loader = torch.utils.data.DataLoader(
        dset_train, batch_size=batch_size, shuffle=True
    )

    model.train()
    for epoch in range(num_epochs):

        epoch_loss = AverageMeter()
        for batch_idx, batch in enumerate(train_loader):
            x, y = batch
            w_optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward(retain_graph=True)

            epoch_loss.update(loss.item())
            if optim == "newton":
                linear_weight = list(model.weight_params())[0]
                hessian_newton = torch.inverse(
                    hessian(loss * 1, linear_weight, linear_weight).reshape(
                        linear_weight.size()[1], linear_weight.size()[1]
                    )
                )
                with torch.no_grad():
                    for w in model.weight_params():
                        w = w.subtract_(torch.matmul(w.grad, hessian_newton))
            elif optim =="sgd":
                torch.nn.utils.clip_grad_norm_(model.weight_params(), 1)
                w_optimizer.step()
            else:
                raise NotImplementedError
        
            wandb.log(
                {"Train loss": epoch_loss.avg, "Epoch": epoch, "Batch": batch_idx}
            )

            if batch_idx % logging_freq == 0:
                print(
                    "Epoch: {}, Batch: {}, Loss: {}, Alphas: {}".format(
                        epoch, batch_idx, epoch_loss.avg, model.fc1.alphas.data
                    )
                )
