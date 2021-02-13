import torch
import torch.nn as nn

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
