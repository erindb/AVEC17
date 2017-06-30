#! /usr/bin/env python

"""
Warmup model for preparing for AVEC17 challenge.
3-class classifier on Desmond's emotion rating data.
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import re

np.random.seed(123)

# ========== read in data ==========

data_dir = "../../data/selfDisclosure"
filenames = os.listdir(data_dir)

data = None

for filename in filenames:
    if re.match("[A-Z0-9]{4}_[0-9]{3}\.csv", filename):
        file_data = np.genfromtxt(os.path.join(data_dir, filename),
                                      delimiter=',',
                                      skip_header=1)
        if data is None:
            data = file_data
        else:
            np.append(data, file_data)

## first dimension is time, then rating
inputs = data[:,2:]
ratings = data[:,1]

def discretize(r):
    if r < 50:
        return "low"
    elif r == 50:
        return "neither"
    elif r > 50:
        return "high"
    else:
        print("error 203948")

classes = ["low", "neither", "high"]

# make one-hot vectors for each output
labels = np.array([classes.index(discretize(r)) for r in ratings])

input_dim = inputs.shape[1]
n_datapoints = inputs.shape[0]

assert(n_datapoints==labels.shape[0])

# ========== create datasets for k-fold cross-validation ==========

K = 5

indices = np.array(range(0, n_datapoints))
np.random.shuffle(indices)

folds = np.split(indices, K)

# ========== set up classifier ==========

# 3 hidden layers with nonlinearities and one linear projection layer

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.w1 = nn.Linear(input_dim, 100)
        self.w2 = nn.Linear(100, 50)
        self.w3 = nn.Linear(50, 20)
        self.w4 = nn.Linear(20, 3)

    def forward(self, x):
        x = F.sigmoid(self.w1(x))
        x = F.sigmoid(self.w2(x))
        x = F.sigmoid(self.w3(x))
        x = self.w4(x)
        return x

net = Net()

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# ========== for each fold: ==========

accuracies = []

for k in range(0, K):
    valid_indices = folds[k]
    train_indices = np.concatenate(folds[0:k] + folds[k+1:])

# ==================== ... train classifier ==========

    n_epochs = 10
    batch_size = 9
    n_batches = int(len(train_indices)/9)

    for epoch in range(n_epochs): 

        running_loss = 0.0

        for b in range(n_batches):

            batch_indices = train_indices[b*batch_size:(b+1)*batch_size]
            # nSamples x nChannels
            train_inputs = inputs[batch_indices,:]
            train_labels = labels[batch_indices,]

            # wrap them in Variable
            train_inputs = Variable(torch.FloatTensor(train_inputs))
            train_labels = Variable(torch.LongTensor(train_labels))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(train_inputs)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if b % 20 == 19:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, b + 1, running_loss / 2000))
                running_loss = 0.0

# ==================== ... compute accuracy ==========

    correct = 0
    total = 0
    valid_inputs = inputs[valid_indices,:]
    valid_labels = labels[valid_indices,]
    
    valid_inputs = Variable(torch.FloatTensor(valid_inputs))
    outputs = net(valid_inputs)
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.numpy().T[0]

    total += valid_labels.shape[0]
    correct += (predicted == valid_labels).sum()

    accuracy = float(correct) / total
    accuracies.append(accuracy)

# ========== compute overall accuracy ==========

print(np.mean(accuracies))
