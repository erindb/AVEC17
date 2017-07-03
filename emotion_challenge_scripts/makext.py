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
import load_features


# Returns a string of the filename given the participant number
def make_name(pNum, split="train"):
    pStr = str(pNum)
    if pNum < 10:
        pStr = '0' + pStr
    return 'Train_' + pStr + '.csv'


# Gets the row index of the time in the the csv file
def time_to_timestep(time, reader, f):
    reader.next()
    step = float(reader.next()[1])  # Step is the second time stamp
    f.seek(0)
    return math.floor(time / step)  # Rounding down

def timestep_to_time(timestep, length_of_timestep):
    return timestep*length_of_timestep


"""
Makes x(t) by taking all features and concatenating them into 1 torch Tensor.

timestep - Integer
pNum - subject number (starts at 1)
dataset - data directory for specific challenge
split - train, valid, test
length_of_timestep - in ms
"""
def make_xt(timestep, pNum, dataset, split="train", length_of_timestep=100):
    assert(split in ["train", "valid", "test"])

    time = timestep_to_time(timestep, length_of_timestep)

    xt = np.float64([])
    folders = [folder for folder in os.listdir(dataset) if 'feature' in folder]
    for folder in folders:
        fname = make_name(pNum, split)
        fullName = os.path.join(dataset, folder, fname)
        with open(fullName, 'r') as file:
            reader = csv.reader(file, delimiter=';')
            ind = time_to_timestep(time, reader, file)
            xt = np.append(xt, np.float64(next(itertools.islice(reader, ind, ind+1))[2:]))
    return torch.from_numpy(xt)


# fix me!!!
def get_num_timepoints(pNum, dataset, split="train", length_of_timestep=100):
    return None

def run_tests():
    assert(1756==get_num_timepoints(1, "../data/AVEC_17_Emotion_Sub-Challenge"))
run_tests()

# Runs t = 0, participant 1 as an example
if __name__ == '__main__':
    print(make_xt(0, 1, "../data/AVEC_17_Emotion_Sub-Challenge"))
    print(get_num_timepoints(1, "../data/AVEC_17_Emotion_Sub-Challenge"))

