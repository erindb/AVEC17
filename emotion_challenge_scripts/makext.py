#!/usr/bin/env python

"""
Loads in data...
"""

__author__      = "Harry Sha"

import os
import numpy as np
import torch
import csv
import math
import itertools


# Returns a string of the filename given the participant number
def make_name(pNum):
    pStr = str(pNum)
    if pNum < 10:
        pStr = '0' + pStr
    return 'Train_' + pStr + '.csv'


# Gets the row index of the time in the the csv file
def get_index(t, reader, f):
    reader.next()
    step = float(reader.next()[1])  # Step is the second time stamp
    f.seek(0)
    return math.floor(t / step)  # Rounding down


# Makes x(t) by taking all features and concatenating them into 1 torch Tensor.
def make_xt(time, pNum, dataset):
    xt = np.float64([])
    folders = [folder for folder in os.listdir(dataset) if 'feature' in folder]
    for folder in folders:
        fname = make_name(pNum)
        fullName = os.path.join(dataset, folder, fname)
        with open(fullName, 'r') as file:
            reader = csv.reader(file, delimiter=';')
            ind = get_index(time, reader, file)
            xt = np.append(xt, np.float64(next(itertools.islice(reader, ind, ind+1))[2:]))
    return torch.from_numpy(xt)


# Runs t = 0, participant 1 as an example
if __name__ == '__main__':
    print(make_xt(0, 1))
