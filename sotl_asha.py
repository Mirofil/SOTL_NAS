from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Experiment
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger
from ray.tune.integration.wandb import wandb_mixin
from ray.tune import Stopper
from typing import *
from collections import defaultdict

def load_data(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset

class CNN(nn.Module):
    def __init__(self, rnorm_scale, rnorm_power):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=3, alpha=rnorm_scale, beta=rnorm_power, k=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(stride=2,kernel_size=3),
            nn.LocalResponseNorm(size=3, alpha=rnorm_scale, beta=rnorm_power, k=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(stride=2,kernel_size=3))

        self.fc1 = nn.Linear(576, 10)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 576)
        x = self.fc1(x)
        return x

class Net(nn.Module):
    def __init__(self, rnorm_scale, rnorm_power):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2) #RELU
        self.rnorm1 = nn.LocalResponseNorm(size=3, alpha=rnorm_scale, beta=rnorm_power, k=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2) # RELU
        self.pool2 = nn.AvgPool2d(stride=2,kernel_size=3)
        self.rnorm2 = nn.LocalResponseNorm(size=3, alpha=rnorm_scale, beta=rnorm_power, k=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2) # RELU
        self.pool3 = nn.AvgPool2d(stride=2,kernel_size=3)

        self.fc1 = nn.Linear(576, 10)

    def forward(self, x):
        x = self.rnorm1(self.pool1(F.relu(self.conv1(x))))
        x = self.rnorm2(self.pool2(F.relu(self.conv2(x))))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 576)
        x = self.fc1(x)
        return x

@wandb_mixin
def train_cifar(config, checkpoint_dir=None, data_dir=None, lr_reductions=True, weight_decay=True):
    net = Net(rnorm_scale=config["rnorm_scale"], rnorm_power=config["rnorm_power"])
    if data_dir is None:
        data_dir = config["data_dir"]

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=1)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=1)
    
    lr_reduction_epochs = [int((config["max_num_epochs"]*config["steps_per_epoch"])/(config["lr_reductions"]+1)*(i+1)) for i in range(config["lr_reductions"])]

    for epoch in range(config["max_num_epochs"]):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0

        for i, data in enumerate(trainloader, 0):
            if lr_reductions and len(lr_reduction_epochs) > 0 and epoch*config["steps_per_epoch"]+i > lr_reduction_epochs[0]:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']/10
                lr_reduction_epochs = lr_reduction_epochs[1:]
            if i > config["steps_per_epoch"]:
                break

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            reg = 0
            if weight_decay:
                for layer, coef in zip([net.conv1, net.conv2, net.conv3, net.fc1], [config["conv1_l2"], config["conv2_l2"], config["conv3_l2"], config["fc1_l2"]]):
                    for name, w in layer.named_parameters():
                        if 'bias' not in name:
                            reg += (w.norm(2)**2 * coef)
            loss = loss + reg
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 200 == 199:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total, lr = optimizer.param_groups[0]['lr'])
    print("Finished Training")

def test_accuracy(net, device="cpu"):
    trainset, testset = load_data(os.path.abspath("../playground/data"))

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=1)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

class TotalBudgetStopper(Stopper):
    """Stop trials after reaching a maximum number of iterations

    Args:
        max_iter (int): Number of iterations before stopping a trial.
    """

    def __init__(self, total_budget: int):
        self._total_budget = total_budget
        self._iter = defaultdict(lambda: 0)

    def __call__(self, trial_id: str, result: Dict):
        self._iter[trial_id] += 1
        total_count = 0
        for trial in self._iter.keys():
            total_count += self._iter[trial]
        return total_count >= self._total_budget

    def stop_all(self):
        total_count = 0
        for trial in self._iter.keys():
            total_count += self._iter[trial]
        return total_count >= self._total_budget

def main(num_samples=16, gpus_per_trial=1):
    data_dir = os.path.abspath("../playground/data")
    load_data(data_dir)  # Download data for all trials before starting the run

    config = {
        "lr": tune.loguniform(5e-5, 5),
        "conv1_l2": tune.loguniform(5e-5, 5),
        "conv2_l2": tune.loguniform(5e-5, 5),
        "conv3_l2": tune.loguniform(5e-5, 5),
        "fc1_l2": tune.loguniform(5e-3, 500),
        "lr_reductions":tune.choice([0,1,2,3]),
        "rnorm_scale": tune.loguniform(5e-6, 5),
        "rnorm_power": tune.uniform(0.01, 3),
        "max_num_epochs":15,
        "batch_size": 100,
        "steps_per_epoch": 100,
        "data_dir" :data_dir
    }
    scheduler = ASHAScheduler(
        max_t=config["max_num_epochs"],
        grace_period=1,
        reduction_factor=4)

    result = tune.run(
        train_cifar,
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config={**config, "wandb": {
            "project": "SoTL_Cifar",
            "api_key_file": r"C:\Users\kawga\.wandb\nas_key.txt"
        }},
        # local_dir = r"C:\Users\kawga\ray_results\inner_2021-01-16_17-25-31",
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler,
        stop=TotalBudgetStopper(config["max_num_epochs"]),
        loggers=DEFAULT_LOGGERS + (WandbLogger, )
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = Net(rnorm_scale=best_trial.config["rnorm_scale"], rnorm_power=best_trial.config["rnorm_power"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


# from ray.tune import Analysis
# analysis = Analysis(r"C:\Users\kawga\ray_results\train_cifar_2021-01-17_00-22-47")

def test_main(gpus_per_trial=1):
    data_dir = os.path.abspath("../playground/data")
    load_data(data_dir)  # Download data for all trials before starting the run
    config = {
        "lr": 1e-2,
        "conv1_l2": 0,
        "conv2_l2": 0,
        "conv3_l2":0,
        "fc1_l2": 0,
        "lr_reductions":2,
        "rnorm_scale": 0.00005,
        "rnorm_power": 0.75,
        "max_num_epochs":300,
        "batch_size": 100,
        "steps_per_epoch": 100,
        "data_dir":data_dir
    }
    scheduler = ASHAScheduler(
        max_t=config["max_num_epochs"],
        grace_period=1,
        reduction_factor=4)

    result = tune.run(
        train_cifar,
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config={**config, "wandb": {
            "project": "SoTL_Cifar",
            "api_key_file": r"C:\Users\kawga\.wandb\nas_key.txt"
        }} ,
        metric="loss",
        mode="min",
        num_samples=1,
        scheduler=scheduler,
        loggers=DEFAULT_LOGGERS + (WandbLogger, ),
        

    )
    train_cifar(config)
