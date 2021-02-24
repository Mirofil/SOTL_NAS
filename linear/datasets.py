import pandas as pd
import torch
from torchvision import datasets, transforms
from utils import data_generator
from pathlib import Path
import numpy as np
import torch
import os
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
from PIL import Image
from urllib.request import urlopen
import scipy.io
from sklearn.model_selection import train_test_split

def get_datasets(name, path=None, val_split=0.1, test_split=0.2, normalize=True, **kwargs):
    n_classes = None
    n_features=None
    if os.name == 'nt':
        data_path = Path("C:\\Users\\kawga\\Documents\\Oxford\\thesis\\code\\data\\")
    else:
        data_path = Path('./data/')

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

        train_data=torch.tensor(np.array(train_data).astype(int), dtype=torch.float32)

        train_classes= torch.tensor(np.array(train_classes).astype(int)[:,0], dtype=torch.long)
        train_classes[train_classes == -1] = 0

        test_data = torch.tensor(np.array(test_data).astype(int), dtype=torch.float32)

        test_classes= torch.tensor(np.array(test_classes).astype(int)[:,0], dtype=torch.long)
        test_classes[test_classes == -1] = 0

        x_train, x_test = train_data, test_data
        y_train, y_test = train_classes, test_classes
        
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
    elif name == 'isolet':
        x_train = np.genfromtxt(data_path / 'isolet1+2+3+4.data', delimiter = ',', usecols = range(0, 617), encoding = 'UTF-8')
        y_train = np.genfromtxt(data_path / 'isolet1+2+3+4.data', delimiter = ',', usecols = [617], encoding = 'UTF-8')
        x_test = np.genfromtxt(data_path / 'isolet5.data', delimiter = ',', usecols = range(0, 617), encoding = 'UTF-8')
        y_test = np.genfromtxt(data_path / 'isolet5.data', delimiter = ',', usecols = [617], encoding = 'UTF-8')
        
        X = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(np.concatenate((x_train, x_test)))
        x_train = torch.tensor(X[: len(y_train)], dtype=torch.float32)
        x_test = torch.tensor(X[len(y_train):], dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

    
    elif name == 'activity':
        x_train = np.loadtxt(data_path / 'har/train/X_train.txt')
        x_test = np.loadtxt(data_path /  'har/test/X_test.txt')
        y_train = np.loadtxt(data_path / 'har/train/y_train.txt')
        y_test = np.loadtxt(data_path /  'har/test/y_test.txt')
        
        X = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(np.concatenate((x_train, x_test)))
        x_train = torch.tensor(X[: len(y_train)], dtype=torch.float32)
        x_test = torch.tensor(X[len(y_train):], dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

    elif name=="madelon":
        train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
        val_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.data'
        train_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
        val_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/madelon_valid.labels'
        test_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_test.data'
        x_train = torch.tensor(np.loadtxt(urlopen(train_data_url)))
        y_train = torch.tensor(np.loadtxt(urlopen(train_resp_url)), dtype=torch.long)
        x_test =  torch.tensor(np.loadtxt(urlopen(val_data_url)))
        y_test =  torch.tensor(np.loadtxt(urlopen(val_resp_url)), dtype=torch.long)

        # TODO finish
    
    elif name == "coil":
        samples = []
        for i in range(1, 21):
            for image_index in range(72):
                obj_img = Image.open(data_path / 'coil-20-proc' / f'obj{i}__{image_index}.png')
                rescaled = obj_img.resize((20,20))
                pixels_values = [float(x) for x in list(rescaled.getdata())]
                sample = np.array(pixels_values + [i])
                samples.append(sample)
        samples = np.array(samples)
        np.random.shuffle(samples)
        data = samples[:, :-1]
        targets = (samples[:, -1] + 0.5).astype(np.int64)
        data = (data - data.min()) / (data.max() - data.min())
        
        l = data.shape[0] * 4 // 5

        x_train, y_train = torch.tensor(data[:l], dtype=torch.float), torch.tensor(targets[:l], dtype=torch.long)
        x_test, y_test = torch.tensor(data[l:], dtype=torch.float), torch.tensor(targets[l:], dtype=torch.long)

    elif name == "MNIST" or name == "MNISTsmall":
        dset_train = datasets.MNIST('./data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

        dset_test = datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        
        if name == "MNISTsmall":
            # This should replicate the MNIST version from the Concrete Autoencoder
            dset_train, dset_test = torch.utils.data.random_split(
                dset_test, [int(len(dset_test) * 0.6), len(dset_test) - int(len(dset_test) * 0.6)]
            )
        
        n_classes = 10
        n_features=28*28

    elif name.lower() == 'relathe':
        mat = scipy.io.loadmat(data_path / "RELATHE.mat")
        x_train, x_test, y_train, y_test = train_test_split(mat['X'], mat['Y'], test_size=test_split)
        mat['Y'] = mat['Y'].flatten()

        n_classes = 2
    elif name.lower() == "basehock":
        mat = scipy.io.loadmat(data_path / "BASEHOCK.mat")
        mat['Y'] = mat['Y'].flatten()
        x_train, x_test, y_train, y_test = train_test_split(mat['X'], mat['Y'], test_size=test_split)
        
        n_classes = 2
    elif name.lower() == "pcmac":
        mat = scipy.io.loadmat(data_path / "PCMAC.mat")
        mat['Y'] = mat['Y'].flatten()

        x_train, x_test, y_train, y_test = train_test_split(mat['X'], mat['Y'], test_size=test_split)
        
        n_classes = 2
    elif name.lower() == "prostate_ge":
        mat = scipy.io.loadmat(data_path / "Prostate_GE.mat")
        mat['Y'] = mat['Y'].flatten()
        x_train, x_test, y_train, y_test = train_test_split(mat['X'], mat['Y'], test_size=test_split)
        
        n_classes = 2
    elif name.lower() == "allaml":
        mat = scipy.io.loadmat(data_path / "ALLAML.mat")
        mat['Y'] = mat['Y'].flatten()
        x_train, x_test, y_train, y_test = train_test_split(mat['X'], mat['Y'], test_size=test_split)

        n_classes = 2
    elif name.lower() == "cll_sub":
        mat = scipy.io.loadmat(data_path / "CLL_SUB_111.mat")
        mat['Y'] = mat['Y'].flatten()
        x_train, x_test, y_train, y_test = train_test_split(mat['X'], mat['Y'], test_size=test_split)

        n_classes = 3
    elif name.lower() == "glioma":
        mat = scipy.io.loadmat(data_path / "GLIOMA.mat")
        mat['Y'] = mat['Y'].flatten()
        x_train, x_test, y_train, y_test = train_test_split(mat['X'], mat['Y'], test_size=test_split)
    
        n_classes = 4
    elif name.lower() == "smk_can":
        mat = scipy.io.loadmat(data_path / "SMK_CAN_187.mat")
        mat['Y'] = mat['Y'].flatten()
        x_train, x_test, y_train, y_test = train_test_split(mat['X'], mat['Y'], test_size=test_split)

        n_classes=2

    elif name == 'MNIST35':
        
        x_train, y_train = load_svmlight_file(str(data_path / 'mnist-35-noisy-binary.train.svm'))
        x_test, y_test = load_svmlight_file(str(data_path / 'mnist-35-noisy-binary.test.svm'))

        x_train = torch.tensor([np.array(elem.todense()) for elem in x_train], dtype=torch.float)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_train[y_train==-1]=0

        x_test = torch.tensor([np.array(elem.todense()) for elem in x_test], dtype=torch.float)
        y_test = torch.tensor(y_test, dtype=torch.long)
        y_test[y_test==-1]=0

        n_classes = 2
        n_features=28*28

    elif name == "FashionMNIST" or name == "FashionMNISTsmall":
        dset_train = datasets.FashionMNIST('./data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

        dset_test = datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        
        if name == "FashionMNISTsmall":
            # This should replicate the MNIST version from the Concrete Autoencoder
            dset_train, dset_test = torch.utils.data.random_split(
                dset_test, [int(len(dset_test) * 0.6), len(dset_test) - int(len(dset_test) * 0.6)]
            )
        
        
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

    if name not in ['CIFAR', 'MNIST', 'FashionMNIST', 'MNISTsmall', 'FashionMNISTsmall']:
        # The datasets from Torchvision have different Classes and they come already well split
       
        if normalize:
            scaler = preprocessing.StandardScaler().fit(x_train)
            x_train = torch.tensor(scaler.transform(x_train), dtype=torch.float32)
            x_test= torch.tensor(scaler.transform(x_test), dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)
        
        dset_train = torch.utils.data.TensorDataset(x_train, y_train)
        dset_test = torch.utils.data.TensorDataset(x_test, y_test)


    dset_train, dset_val = torch.utils.data.random_split(
            dset_train, [int(len(dset_train) * (1-val_split)), len(dset_train) - int(len(dset_train) * (1-val_split))]
        )
    if name in ['MNIST', 'CIFAR', "FashionMNIST", 'isolet', 'madelon', 'activity', 'coil']:
        task = 'multiclass'
    elif name in ['gisette', 'MNIST35']:
        task = 'binary'
    else:
        task = 'reg'

    if n_classes is None:
        if len(dset_train[0][1].size()) == 0:
            n_classes = 1
        else:
            n_classes = dset_train[0][1].size()[1]

    if n_features is None:
        n_features = dset_train[0][0].view(1, -1).shape[1]

    results = {}
    results['dset_train'] = dset_train
    results['dset_val'] = dset_val
    results['dset_test'] = dset_test
    results['task'] = task
    results['n_classes'] = n_classes
    results['n_features'] = n_features

    assert all([elem != None for elem in results.values()])

    return results
