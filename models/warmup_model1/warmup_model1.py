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

labels = np.array([classes.index(discretize(r)) for r in ratings])

input_dim = inputs.shape[1]
n_datapoints = inputs.shape[0]

assert(n_datapoints==labels.shape[0])

## zip labels and inputs together?

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
        x = self.w1(F.Sigmoid(self.w1(x)))
        x = self.w2(F.Sigmoid(self.w2(x)))
        x = self.w3(F.Sigmoid(self.w3(x)))
        x = self.w4(x)
        return np.argmax(x)

net = Net()

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# ========== for each fold: ==========

# a = np.arange(9, -1, -1)     # a = array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
# b = a[np.arange(len(a))!=3]  # b = array([9, 8, 7, 5, 4, 3, 2, 1, 0])
# which will, in general, be much faster than the list comprehension listed above.

for k in range(0, K):
    valid_indices = folds[k]
    train_indices = np.concatenate(folds[0:k] + folds[k+1:])

# ==================== ... train classifier ==========

# ==================== ... compute accuracy ==========

# ========== compute overall accuracy ==========
