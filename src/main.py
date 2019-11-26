import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

linear_X(X_all, Y, features, labels, label_names, isTrain, datasetPath)
#linear_one_hot(X_all, Y, features, labels, label_names, isTrain, datasetPath)