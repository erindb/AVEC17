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

# ========== read in datapoints ==========

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

labels = [classes.index(discretize(r)) for r in ratings]

input_dim = inputs.shape[1]

## convert to torch variables...
# inputs, labels = ...

# ========== create datasets for k-fold cross-validation ==========

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

# ==================== ... train classifier ==========

# ==================== ... compute accuracy ==========

# ========== compute overall accuracy ==========
