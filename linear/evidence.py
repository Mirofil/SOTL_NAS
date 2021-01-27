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

class EvidenceNet(nn.Module):

    def __init__(self, D, N, alpha_1, alpha_2, beta_1, beta_2, C, c):
        super().__init__()
        self.D=torch.tensor(D)
        self.N=torch.tensor(N)
        self.alpha_1=torch.tensor(alpha_1)
        self.beta_1=torch.tensor(beta_1)
        self.alpha_2=torch.tensor(alpha_2)
        self.beta_2=torch.tensor(beta_2)
        self.C = nn.Parameter(C, requires_grad=True)
        self.c = torch.tensor(c, requires_grad=True)

    def forward(self, x, y):
        return evidence_log(x,y, self.C, self.c, self.alpha_1, self.alpha_2, self.beta_1, self.beta_2, self.N)

##### EVIDENCE MAXIMIZATION

D=torch.tensor(3)
N=torch.tensor(5000)
alpha_1=torch.tensor(1.)
beta_1=torch.tensor(1.)
alpha_2=torch.tensor(1.)
beta_2=torch.tensor(1.)
C=torch.tensor(3.0, requires_grad=True)
design_matrix = torch.tensor(np.random.rand(N, 2*D-1), dtype=torch.float32)
c = torch.tensor([1.0 for _ in range(D)], requires_grad=True)
y = torch.tensor(np.random.rand(N,1))

tau = sample_tau(alpha_1, beta_1)

# define the number of epochs and the data set size
nb_epochs = 5000
model = EvidenceNet(D=2, N=N, alpha_1=alpha_1, alpha_2=alpha_2, beta_1=beta_1, beta_2=beta_2, C=C, c=c)
optimizer = SGD(model.parameters(), lr=0.01)

x_train, y_train = data_generator(N, max_order=D, noise_var=1/float(tau), featurize_type='fourier')
# create our training loop
for epoch in range(nb_epochs):

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    dset = torch.utils.data.TensorDataset(x_train, y_train)

    train_loader = torch.utils.data.DataLoader(dset, batch_size = int(N))
    epoch_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        x, y = batch
        epoch_loss = 0

        evidence = model(x, y)
        loss = -evidence

        epoch_loss += loss.data
        optimizer.zero_grad()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        loss.backward()

        optimizer.step()
        if batch_idx % 200 == 0:
            print("Epoch: {}, Batch: {}, Loss: {}, C: {}".format(epoch, batch_idx, epoch_loss, model.C.item()))

# test the model
model.eval()
test_data = data_generator(1)
prediction = model(Variable(Tensor(test_data[0][0])))
print("Prediction: {}".format(prediction.data[0]))
print("Expected: {}".format(test_data[1][0]))

test = next(iter((train_loader)))