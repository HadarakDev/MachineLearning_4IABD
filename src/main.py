import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from linear_X import linear_X
from linear_one_hot import linear_one_hot
from tools import unpickle, get_label_names, display_batch_stat

datasetPath = "../dataset/data_batch_"
size = 32
isTrain = True

label_names = get_label_names()
#for i in range(1, 5):
#    display_batch_stat(i, label_names, datasetPath)

features, labels = unpickle("../dataset/data_batch_1", size, True)

X_all = features.flatten().reshape(10000, size * size * 3)
Y = np.asarray(labels)


data = pd.read_csv("../config/linear_X.csv") 
for i in range(data.shape[0]):
    activation_param = data['activation_param'].iloc[i]
    optimizer_param = data['optimizer_param'].iloc[i]
    loss_param = data['loss_param'].iloc[i]
    batch_size_param = data['batch_size_param'].iloc[i]
    epochs_param = data['epochs_param'].iloc[i]
    save_path_info = data['save_path_info'].iloc[i]
    print("START NEW TRAINING")
    print("activation_param : " + str(activation_param))
    print("optimizer_param : " + str(optimizer_param))
    print("loss_param : " + str(loss_param))
    print("batch_size_param : " + str(batch_size_param))
    print("epochs_param : " + str(epochs_param))
    print("save_path_info : " + str(save_path_info))
    linear_X(X_all, Y, isTrain, activation_param, optimizer_param, loss_param, batch_size_param, epochs_param, save_path_info)
#linear_one_hot(X_all, Y, features, labels, label_names, isTrain, datasetPath)