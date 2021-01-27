import torch
from torch import Tensor
from torch.nn import Linear, MSELoss, functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import math
import itertools
from utils import evidence_log, evidence, c_tilde, sample_tau, sample_C, c_cov, sample_w, featurize, eval_features, data_generator

# define the model
class Net(torch.nn.Module):
    def __init__(self, num_features = 2):
        super(Net, self).__init__()
        self.fc1 = Linear(num_features, 1, bias=False)

    def forward(self, x):
        return self.fc1(x)


# define the number of epochs and the data set size
nb_epochs = 500

D=torch.tensor(3)
N=torch.tensor(5000)

model = Net(num_features=D)

criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

x_train, y_train = data_generator(N, max_order=D, noise_var=0, featurize_type='polynomial')

for epoch in range(nb_epochs):

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    dset = torch.utils.data.TensorDataset(x_train, y_train)

    train_loader = torch.utils.data.DataLoader(dset, batch_size = 64)
    epoch_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        x, y = batch

        epoch_loss = 0

        y_pred = model(x)
        loss = criterion(y_pred, y)

        epoch_loss += loss.data
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        if batch_idx % 200 == 0:
            print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch_idx, epoch_loss))

# test the model
model.eval()
test_data = data_generator(1)
prediction = model(Variable(Tensor(test_data[0][0])))
print("Prediction: {}".format(prediction.data[0]))
print("Expected: {}".format(test_data[1][0]))

test = next(iter((train_loader)))
