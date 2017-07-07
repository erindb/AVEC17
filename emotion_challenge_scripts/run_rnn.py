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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sys     import argv
from sklearn import svm

from load_features     import load_all
from calc_scores       import calc_scores
from write_predictions import write_predictions

from os.path import join as pjoin
from makeXs import make_Xs, get_num_timesteps

# ================= Load features ================= 

# Set folders here
path_test_predictions = "test_predictions/"
b_test_available      = False  # If the test labels are not available, the predictions on test are written into the folder 'path_test_predictions'
data_dir = "../data/AVEC_17_Emotion_Sub-Challenge"

hidden_size = 10 #4000
h2_size = 10 #50
batch_size = 1
num_epochs = 2

torch.manual_seed(123)

# # random input
# X_np = np.random.rand(1756, 1, 8962)


class DataLoader():
    def __init__(self, pNum, data_dir, split="train", length_of_timestep=100,
        useAudio=True, useVideo=True, useText=True):
        self.labelDict = {'arousal': 1, 'valence': 2, 'liking': 3}
        self.Y_np, self.X_np = make_Xs(pNum=pNum, data_dir=data_dir, split=split,
                             length_of_timestep=length_of_timestep,
                             useAudio=useAudio, useVideo=useVideo,
                             useText=useText)

    def read_data(self, labelType):
        # label_num = 1, 2 or 3.
        assert (labelType in self.labelDict)
        
        label_num = self.labelDict[labelType]

        # X = np.random.rand(1756, 1, 8962)
        # exclude first column (time)
        X_np = self.X_np[:, 1:]
        Y_np = self.Y_np[:, label_num]

        X_np = X_np.reshape(X_np.shape[0], 1, X_np.shape[1])
        Y_np = Y_np.reshape(Y_np.shape[0], 1, 1)

        seq_len = X_np.shape[0]
        num_features = X_np.shape[2]

        X_tensor = torch.from_numpy(X_np).float()
        X = Variable(X_tensor)

        # Y_np = np.random.rand(1756, 1, 1) # do later
        Y_tensor = torch.from_numpy(Y_np).float()
        Y = Variable(Y_tensor)

        return X, Y, num_features, seq_len


# real input
trainLoader = DataLoader(pNum = 1, data_dir=data_dir, useAudio=True, 
    useVideo=False, useText=False)
X, Y, num_features, seq_len = trainLoader.read_data('arousal')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn1 = nn.GRU(input_size=num_features,
                           hidden_size=hidden_size,
                           num_layers=1)
        self.W1 = nn.Linear(hidden_size, h2_size)
        self.W2 = nn.Linear(h2_size, 1)


    def forward(self, x, hidden):
        x, hidden = self.rnn1(x, hidden)

        h1 = x.view(-1, hidden_size)
        h2 = F.relu(self.W1(h1))

        y = self.W2(h2)

        return y, hidden

    def init_hidden(self, batch_size=batch_size):
        weight = next(self.parameters()).data
        return Variable(weight.new(1, batch_size, hidden_size).zero_())
        # return Variable(torch.from_numpy(np.random.rand(1,1,hidden_size)))

labelTypes = ["arousal", "valence", "liking"]
models = {labelType: Net() for labelType in labelTypes}

criterion = nn.MSELoss()

# ================= Run our model ================= 

#...
def train(labelType):
    X, Y, num_features, seq_len = trainLoader.read_data(labelType)

    model = models[labelType]
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    hidden = model.init_hidden()
    # for each batch

    for epoch in range(num_epochs):
        model.zero_grad()
        output, hidden = model.forward(X, hidden)
        loss = criterion(output, Y)
        print ('Epoch [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, loss.data[0]))        
        loss.backward(retain_variables=True)
        optimizer.step()

for labelType in labelTypes:
    train(labelType)

# try this?:
# https://discuss.pytorch.org/t/rnn-for-sequence-prediction/182/15

testLoader = DataLoader(pNum = 1, data_dir=data_dir, split="devel",
                        useAudio=True, useVideo=False, useText=False)

def test(labelType):
    X, Y, num_features, seq_len = testLoader.read_data(labelType)

    model = models[labelType]

    hidden = model.init_hidden()

    output, hidden = model.forward(X, hidden)
    return calc_scores(output.data.numpy(), Y.data.numpy())


## fix me!!!
# this is the basic format of how the output should look:

# arousal scores
scores_devel_A = np.array([test("arousal")])
# valence scores
scores_devel_V = np.array([test("valence")])
# liking scores
scores_devel_L = np.array([test("liking")])

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
