#!/bin/python
# python2.7
# Train and evaluate (audio-visual) model for the prediction of arousal / valence / liking
# 
# The performance on the development set in terms of CCC, PCE, and MSE is appended to corresponding text files (results.txt, results_pcc.txt, results_mse.txt).
# The predicitions on the test set are written into the folder specified by the variable 'path_test_predictions'.
# 
# Arguments (all optional):
#  argv[1]: Delay (in seconds, shift features to the back to compensate delay of annotations)
#  argv[2] argv[3] argv[4]: Each of these arguments is either 0 or 1 and determines if audio, video, and/or text features are taken into account
#  Default is: 0.0 1 1 1 
# 
# modified from run_baseline.py (Contact: maximilian.schmitt@uni-passau.de)

import os
import fnmatch
import numpy as np

from sys     import argv
from sklearn import svm

from load_features     import load_all
from calc_scores       import calc_scores
from write_predictions import write_predictions

from os.path import join as pjoin

from makext import make_name, get_index, make_xt

# ================= Load features ================= 

# Set folders here
path_test_predictions = "test_predictions/"
b_test_available      = False  # If the test labels are not available, the predictions on test are written into the folder 'path_test_predictions'
data_dir = "../data/AVEC_17_Emotion_Sub-Challenge"

print(make_xt(0, 1, data_dir))

# ================= Run our model ================= 

# try this?:
# https://discuss.pytorch.org/t/rnn-for-sequence-prediction/182/15






## fix me!!!
# this is the basic format of how the output should look:

# arousal scores
scores_devel_A = np.array([[None, None, None], [None, None, None]])
# valence scores
scores_devel_V = np.array([[None, None, None], [None, None, None]])
# liking scores
scores_devel_L = np.array([[None, None, None], [None, None, None]])

## what follows is unchanged from baseline

ind_opt_A = np.argmax(scores_devel_A[:,0])
ind_opt_V = np.argmax(scores_devel_V[:,0])
ind_opt_L = np.argmax(scores_devel_L[:,0])

#  ================= Print scores  ================= 
# (CCC, PCC, RMSE) on the development set for our model

print("Arousal devel (CCC,PCC,RMSE):")
print(scores_devel_A[ind_opt_A,:])
print("Valence devel (CCC,PCC,RMSE):")
print(scores_devel_V[ind_opt_V,:])
print("Liking  devel (CCC,PCC,RMSE):")
print(scores_devel_L[ind_opt_L,:])

if b_test_available:
    result_ccc  = [ scores_devel_A[ind_opt_A,0], score_test_A[0], scores_devel_V[ind_opt_V,0], score_test_V[0], scores_devel_L[ind_opt_L,0], score_test_L[0] ]
    result_pcc  = [ scores_devel_A[ind_opt_A,1], score_test_A[1], scores_devel_V[ind_opt_V,1], score_test_V[1], scores_devel_L[ind_opt_L,1], score_test_L[1] ]
    result_rmse = [ scores_devel_A[ind_opt_A,2], score_test_A[2], scores_devel_V[ind_opt_V,2], score_test_V[2], scores_devel_L[ind_opt_L,2], score_test_L[2] ]
    print("Arousal test (CCC,PCC,RMSE):")
    print(score_test_A)
    print("Valence test (CCC,PCC,RMSE):")
    print(score_test_V)
    print("Liking  test (CCC,PCC,RMSE):")
    print(score_test_L)
else:
    # Write only the scores for the development set
    result_ccc  = [ scores_devel_A[ind_opt_A,0], scores_devel_V[ind_opt_V,0], scores_devel_L[ind_opt_L,0] ]
    result_pcc  = [ scores_devel_A[ind_opt_A,1], scores_devel_V[ind_opt_V,1], scores_devel_L[ind_opt_L,1] ]
    result_rmse = [ scores_devel_A[ind_opt_A,2], scores_devel_V[ind_opt_V,2], scores_devel_L[ind_opt_L,2] ]

# Write scores into text files
with open("results_ccc.txt", 'a') as myfile:
    myfile.write("Arousal Valence Liking\n")
    myfile.write(str(result_ccc) + '\n')
with open("results_pcc.txt", 'a') as myfile:
    myfile.write("Arousal Valence Liking\n")
    myfile.write(str(result_pcc) + '\n')
with open("results_rmse.txt", 'a') as myfile:
    myfile.write("Arousal Valence Liking\n")
    myfile.write(str(result_rmse) + '\n')
