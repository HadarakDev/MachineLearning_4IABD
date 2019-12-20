import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from linear_X import linear_X
from linear_one_hot import linear_one_hot
from linear_sparse import linear_sparse
from nn import nn_one_hot
from cnn import test
from tools import unpickle, get_label_names, display_batch_stat, create_dirs


datasetPath = "../dataset/data_batch_"
size = 32
isTrain = True

label_names = get_label_names()
#for i in range(1, 5):
#    display_batch_stat(i, label_names, datasetPath)

X_all = []
Y = []

for i in range(1, 6):
    features, labels = unpickle("../dataset/data_batch_{}".format(i), size, True)
    X_all.append(features.flatten().reshape(10000, size * size * 3))
    Y.append(np.asarray(labels))

X_all = np.concatenate(X_all)
#X_all = X_all / 255.0

Y = np.concatenate(Y)

def linear():
    data = pd.read_csv("../config/linear.csv") 
    for i in range(data.shape[0]):
        activation_param = data['activation_param'].iloc[i]
        optimizer_param = data['optimizer_param'].iloc[i]
        lr_param = data['learning_rate'].iloc[i]
        loss_param = data['loss_param'].iloc[i]
        loss_param_sparse = "sparse_" + loss_param
        batch_size_param = data['batch_size_param'].iloc[i]
        epochs_param = data['epochs_param'].iloc[i]
        save_path_info = data['save_path_info'].iloc[i]
        print("START NEW TRAINING")
        print("activation_param : " + str(activation_param))
        print("optimizer_param : " + str(optimizer_param))
        print("learning_rate : " + str(lr_param))
        print("loss_param : " + str(loss_param))
        print("batch_size_param : " + str(batch_size_param))
        print("epochs_param : " + str(epochs_param))
        print("save_path_info : " + str(save_path_info))
        #linear_X(X_all, Y, isTrain, activation_param, optimizer_param, lr_param, loss_param, batch_size_param, epochs_param, save_path_info)
        #linear_one_hot(X_all, Y, isTrain, activation_param, optimizer_param, lr_param, loss_param, batch_size_param, epochs_param, save_path_info)
        
        save_path_info_sparse = save_path_info.split("_")
        save_path_info_sparse.insert(3, "sparse")
        save_path_info_sparse = "_".join(save_path_info_sparse)
        linear_sparse(X_all, Y, isTrain, activation_param, optimizer_param, lr_param, loss_param_sparse, batch_size_param, epochs_param, save_path_info_sparse)

def nn():
    data = pd.read_csv("../config/nn.csv") 
    for i in range(data.shape[0]):
        activation_param = data['activation_param'].iloc[i].split(";")
        optimizer_param = data['optimizer_param'].iloc[i]
        lr_param = data['learning_rate'].iloc[i]
        loss_param = data['loss_param'].iloc[i]
        batch_size_param = data['batch_size_param'].iloc[i]
        epochs_param = data['epochs_param'].iloc[i]
        save_path_info = data['save_path_info'].iloc[i]
        array_layers = data['array_layers'].iloc[i].split(";")
        print("START NEW TRAINING")
        print("activation_param : " + str(activation_param))
        print("optimizer_param : " + str(optimizer_param))
        print("learning_rate : " + str(lr_param))
        print("loss_param : " + str(loss_param))
        print("batch_size_param : " + str(batch_size_param))
        print("epochs_param : " + str(epochs_param))
        print("save_path_info : " + str(save_path_info))
        print("array_layers : " + str(array_layers))
        nn_one_hot(X_all, Y, isTrain, activation_param, optimizer_param, lr_param, loss_param, batch_size_param, epochs_param, save_path_info, array_layers)

# test(X_all, Y)

if __name__ == "__main__":
    create_dirs()
    linear()