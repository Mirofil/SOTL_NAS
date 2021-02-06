import os
import gdown
import shutil
gdrive_torch_home = "/content/drive/MyDrive/Colab Notebooks/data/TORCH_HOME"

if os.path.exists(gdrive_torch_home):
  os.environ["TORCH_HOME"] = "/content/drive/MyDrive/Colab Notebooks/data/TORCH_HOME"
  nats_bench = "https://drive.google.com/uc?id=17_saCsj_krKjlCBLOJEpNtzPXArMCqxU"
  output = gdrive_torch_home + os.sep+ 'NATS-tss-v1_0-3ffb9-simple.tar'

  gdown.download(url, output, quiet=False)
  shutil.unpack_archive(output, gdrive_torch_home)


import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
[transforms.ToTensor(),
 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='/content/drive/MyDrive/Colab Notebooks/data', train=True,
                                    download=True, transform=transform)
trainset = torchvision.datasets.CIFAR10(root='/content/drive/MyDrive/Colab Notebooks/data', train=False,
                                    download=True, transform=transform)

train = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=64, shuffle=True, **kwargs)
test = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=False,
                   transform=transforms.Compose([transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=64, shuffle=True, **kwargs)