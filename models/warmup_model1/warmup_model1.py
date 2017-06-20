#! /usr/bin/env python

"""
Warmup model for preparing for AVEC17 challenge.
3-class classifier on Desmond's emotion rating data.
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# ========== read in datapoints ==========

input_dim = 1
classes = ["high", "neither", "low"]

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

# ========== for each fold: ==========

# ==================== ... train classifier ==========

# ==================== ... compute accuracy ==========

# ========== compute overall accuracy ==========
