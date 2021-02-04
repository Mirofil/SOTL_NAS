import pandas as pd
import torch
from torchvision import datasets, transforms
from utils import data_generator

def get_datasets(name, path=None, test_split=0.85, **kwargs):
    if name == "songs":
        if path is None:
            path = r"C:\Users\kawga\Documents\Oxford\thesis\data\YearPredictionMSD.txt"

        data = pd.read_csv(path)
        y = data[data.columns[0]].values.tolist()
        x = data[data.columns[1:]].values.tolist()

        y = torch.tensor(y, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)
        dset = torch.utils.data.TensorDataset(x, y)

        dset_train, dset_test = torch.utils.data.random_split(
            dset, [int(len(dset) * test_split), len(dset) - int(len(dset) * test_split)]
        )
    elif name == "fourier":
        x, y = data_generator(
            **kwargs
        )
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        dset = torch.utils.data.TensorDataset(x, y)

        dset_train, dset_test = torch.utils.data.random_split(
            dset, [int(len(dset) * test_split), len(dset) - int(len(dset) * test_split)]
        )
    elif name == "MNIST":
        dset_train = datasets.MNIST('./data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

        dset_test = datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    
    elif name == "CIFAR":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        dset_train = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)

        dset_test = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    
    return dset_train, dset_test
