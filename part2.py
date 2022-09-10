# Creating train and test data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X_train, X_test, y_train, y_test = train_test_split(BoW_human_del, human_y_del, test_size = 0.20, random_state = 123)

train_shape = len(X_train)
test_shape = len(X_test)

# Changing training and testing data to use in model
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

n_samples, n_features = BoW_human.shape
#print(n_samples, n_features)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

y_train = F.one_hot(y_train, num_classes=7)

y_test = F.one_hot(y_test, num_classes=7)

# Functions to use in model
def true_false_tensor(shape,y_pred,y_train):

    y_pred1 = np.zeros(y_pred.shape)
    for i in range(shape):
        val = torch.max(y_pred[i],0)
        y_pred1[i,val.indices.item()] = 1.0

    y_pred2 = torch.from_numpy(y_pred1.astype(np.float32))

    true_false = y_pred2.eq(y_train)

    return true_false, y_pred2


def accuracy(shape,y_pred,y_train):
    
    true_false, y_pred2 = true_false_tensor(shape,y_pred,y_train)
    true_false2 = []

    for i in range(shape):
        true_false1 = []

        for j in range(7):
            if true_false[i,j].item() == False:
                true_false1.append(False)   
            else:
                true_false1.append(True)
        if False in true_false1:
            true_false2.append(False)
        else:
            true_false2.append(True)

    true_false2 = np.array(true_false2)
    true_false2 = torch.from_numpy(true_false2)
    acc_train = true_false2.sum() / y_train.shape[0]
    return acc_train, y_pred2

