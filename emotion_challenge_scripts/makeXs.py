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
    return split.capitalize() + '_' + pStr + '.csv'


# Gets the row index of the time in the the csv file
def seconds_to_timestep(time_in_seconds, reader, f):
    reader.next()
    step = float(reader.next()[1])  # Step is the second time stamp
    f.seek(0)
    return int(math.floor(time_in_seconds / step))  # Rounding down

def timestep_to_seconds(timestep, length_of_timestep):
    return (timestep*length_of_timestep) / 1000


"""
Makes x(t) by taking all features and concatenating them into 1 torch Tensor.

    timestep - Integer
    pNum - subject number (starts at 1)
    dataset - data directory for specific challenge
    split - train, valid, test
    length_of_timestep - in ms
"""
def make_xt(timestep, pNum, dataset, split="train", length_of_timestep=100):
    if timestep % 100 == 0:
        print(timestep)
    assert(split in ["train", "devel", "test"])

    time_in_seconds = timestep_to_seconds(timestep, length_of_timestep)

    xt = np.float64([])
    folders = [folder for folder in os.listdir(dataset) if 'feature' in folder]
    for folder in folders:
        fname = make_name(pNum, split)
        fullName = os.path.join(dataset, folder, fname)
        with open(fullName, 'r') as file:
            reader = csv.reader(file, delimiter=';')
            ind = seconds_to_timestep(time_in_seconds, reader, file)
            xt = np.append(xt, np.float64(next(itertools.islice(reader, ind, ind+1))[2:]))
    return xt

def make_Xs(pNum, data_dir, split="train", length_of_timestep=100,
    useAudio=True, useVideo=True, useText=True):
    '''
    returns labelsT, featuresT

    1st column of labelsT is the time
    2nd column of labelsT is valence [same order as they are in the labels file]
    3rd column of labelsT is arousal [same order as they are in the labels file]
    4th column of labelsT is liking [same order as they are in the labels file]

    1st column of featuresT is the time
    2nd column onwards are all the features

    - written by desmond, bug him if any bugs.
    '''

    # data_dir="../data/AVEC_17_Emotion_Sub-Challenge"
    path_labels = os.path.join(data_dir, "labels/")
    path_features = []
    pNumFilename = make_name(pNum, split)

    if length_of_timestep == 100:
        length_of_timestep = 0.1  # I want this in seconds...

    ### --- reads in label if split is not "test" --- ###
    labelsT = np.float64([])
    if split != 'test':
        fullName = os.path.join(path_labels, pNumFilename)
        with open(fullName, 'r') as thisFile:
            myReader = csv.reader(thisFile, delimiter=';')
            numRows = 0
            for myRow in myReader:
                numRows += 1
                labelsT = np.append(labelsT, np.float64(myRow[1:]), axis=0)
            numLabels = len(myRow) - 1
            assert(numLabels * numRows == len(labelsT))
        labelsT = labelsT.reshape(numRows, numLabels)
        maxTimePoints = labelsT.shape[0]

    if not maxTimePoints:
        maxTimePoints = -1  # reach here if reading from the test set; we don't know how long

    featuresT = np.float64([])

    if useAudio:
        path_features.append( os.path.join(data_dir, "audio_features_xbow_6s/") )
    if useVideo:
        path_features.append( os.path.join(data_dir, "video_features_xbow_6s/") )
    if useText:
        path_features.append( os.path.join(data_dir, "text_features_xbow_6s/") )

    ### --- loop to read in features --- ###
    for thisFolder in path_features:
        print("makeXs: reading in __" + split + "__ features from " + thisFolder)
        theseFeatures = np.float64([])
        fullName = os.path.join(thisFolder, pNumFilename)
        ### --- first, read in ALL rows. Interpolate later --- ###
        with open(fullName, 'r') as thisFile:
            myReader = csv.reader(thisFile, delimiter=';')
            numRows = 0
            for myRow in myReader:
                numRows += 1
                theseFeatures = np.append(theseFeatures, np.float64(myRow[1:]), axis=0)
            numFeatures = len(myRow) - 1
            assert(numFeatures * numRows == len(theseFeatures))
        theseFeatures = theseFeatures.reshape(numRows, numFeatures)
        print("makeXs: ... done reading in features!")

        if not np.isclose(theseFeatures[1, 0] - theseFeatures[0, 0], length_of_timestep):
            # need interpolation.
            ### --- interpolate --- ###
            # use numpy.interp function
            #     newFeatureVector = numpy.interp ( newTimeVector, oldTimeVector, oldFeatureVector )
            totalTimeSteps = int((theseFeatures[-1,0])/length_of_timestep)
            newTimeVector = [float(x)*length_of_timestep for x in xrange(totalTimeSteps)]
            # ... todo: FIX ME
            print("Oops you reached here; interpolation isn't working yet. wah wah.")
            print("let desmond know.")

        if featuresT.shape==(0,):
            featuresT = theseFeatures
        else:
            featuresT = np.append(featuresT, theseFeatures[:, 1:], axis=1)

    # return the labels and the features
    return labelsT, featuresT

"""
Desmond, fix me!!!
"""
def makeX():
    return None

"""
Each participant has a different number of timesteps

    pNum - subject number (starts at 1)
    dataset - data directory for specific challenge
    split - train, valid, test
    length_of_timestep - in ms
"""
def get_num_timesteps(pNum, dataset, split="train", length_of_timestep=100):
    assert(split in ["train", "devel", "test"])

    finest_granularity_filename = os.path.join(dataset,
                                               "audio_features/",
                                               make_name(pNum, split="train"))
    last_line = open(finest_granularity_filename).readlines()[-1]
    max_time_in_seconds = float(last_line.split(";")[1])
    max_time_in_ms = max_time_in_seconds*1000
    num_full_intervals = int(max_time_in_ms) / int(length_of_timestep)
    return num_full_intervals + 1


def run_tests():
    assert(1756==get_num_timesteps(1, "../data/AVEC_17_Emotion_Sub-Challenge"))
run_tests()


# Runs t = 0, participant 1 as an example
if __name__ == '__main__':
    print(make_xt(0, 1, "../data/AVEC_17_Emotion_Sub-Challenge"))
    print(get_num_timesteps(1, "../data/AVEC_17_Emotion_Sub-Challenge"))

