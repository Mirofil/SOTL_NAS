import pandas as pd
import torch
from torchvision import datasets, transforms
from utils import data_generator
from pathlib import Path
import numpy as np
import torch
import os
def get_datasets(name, path=None, test_split=0.85, **kwargs):
    n_classes = 1
    n_features=None
    if name == "songs":
        if path is None:
            path = r"C:\Users\kawga\Documents\Oxford\thesis\data\YearPredictionMSD.txt"

        train_data = pd.read_csv(path)
        y = train_data[train_data.columns[0]].values.tolist()
        x = train_data[train_data.columns[1:]].values.tolist()

        y = torch.tensor(y, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)
        dset = torch.utils.data.TensorDataset(x, y)

        dset_train, dset_test = torch.utils.data.random_split(
            dset, [int(len(dset) * test_split), len(dset) - int(len(dset) * test_split)]
        )
    elif name == "gisette":
        if os.name == 'nt':
            data_path = Path("C:\\Users\\kawga\\Documents\\Oxford\\thesis\\code\\data\\")
        else:
            data_path = Path('./data/')
        f= open(data_path / "gisette_train.data")
        train_data=[]
        for row in f.readlines():
            train_data.append((row.strip()).split(" "))
        f.close()

        f= open(data_path / "gisette_train.labels")
        train_classes=[]
        for row in f.readlines():
            train_classes.append((row.strip()).split(" "))
        f.close()

        f= open(data_path / "gisette_valid.data")
        test_data=[]
        for row in f.readlines():
            test_data.append((row.strip()).split(" "))
        f.close()

        f= open(data_path / "gisette_valid.labels")
        test_classes=[]
        for row in f.readlines():
            test_classes.append((row.strip()).split(" "))
        f.close()

        train_data=np.array(train_data).astype(int)
        train_data = torch.tensor(train_data, dtype=torch.float32)

        train_classes= np.array(train_classes).astype(int)
        train_classes=train_classes[:,0]
        train_classes = torch.tensor(train_classes, dtype=torch.float32)
        train_classes[train_classes == -1] = 0

        test_data=np.array(test_data).astype(int)
        test_data = torch.tensor(test_data, dtype=torch.float32)

        test_classes= np.array(test_classes).astype(int)
        test_classes=test_classes[:,0]
        test_classes = torch.tensor(test_classes, dtype=torch.float32)
        test_classes[test_classes == -1] = 0

        dset_train = torch.utils.data.TensorDataset(train_data, train_classes)
        dset_test = torch.utils.data.TensorDataset(test_data,test_classes)

        n_classes = 2
        n_features=5000
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

        n_classes = 1
        
    elif name == "MNIST":
        dset_train = datasets.MNIST('./data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

        dset_test = datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        n_classes = 10
        n_features=28*28

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
        n_classes = 10
    
    dset_train, dset_val = torch.utils.data.random_split(
            dset_train, [int(len(dset_train) * test_split), len(dset_train) - int(len(dset_train) * test_split)]
        )

    if name in ['MNIST', 'CIFAR', 'gisette']:
        task = 'clf'
    else:
        task = 'reg'

    results = {}
    results['dset_train'] = dset_train
    results['dset_val'] = dset_val
    results['dset_test'] = dset_test
    results['task'] = task
    results['n_classes'] = n_classes
    results['n_features'] = n_features
    # return dset_train, dset_val, dset_test, task, n_classes, n_features
    return results
