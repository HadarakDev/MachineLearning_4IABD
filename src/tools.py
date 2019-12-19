import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

def unpickle(path_batch, size, isRGB):
    with open(path_batch, 'rb') as fo:
        batch = pickle.load(fo, encoding='latin1')
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    # Make a resize here depending on 'size' and 'isRBG'
    labels = batch['labels']
    return features, labels

def get_label_names():
    with open("../dataset/batches.meta", 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return (data['label_names'])

def display_batch_stat(batch_nb, label_names, datasetPath, size, isRGB):
    features, labels = unpickle(datasetPath + str(batch_nb), size, isRGB)
    print("Batch N° %s" % str(batch_nb), "\n")
    print("Number of Samples in batch %s" % str(len(features)), "\n")
    counts =  [[x, labels.count(x)] for x in set(labels)]
    for c in counts:
        print( "%s = %d <=> %.2f %s" % (label_names[c[0]], c[1], (100 * c[1]) / len(features), "%"))

def load_linear_model(model_path):
    return tf.keras.models.load_model(model_path)

def y_one_hot(Y, nb_output):
    return tf.one_hot(Y, nb_output)

def get_optimizer(optimizer_param, lr_param):
    if optimizer_param == "adadelta":
        optimizer_param = Adadelta(lr=lr_param)
    if optimizer_param == "adagrad":
        optimizer_param = Adagrad(lr=lr_param)
    if optimizer_param == "adam":
        optimizer_param = Adam(lr=lr_param)
    if optimizer_param == "adamax":
        optimizer_param = Adamax(lr=lr_param)
    if optimizer_param == "ftrl":
        optimizer_param = Ftrl(lr=lr_param)
    if optimizer_param == "nadam":
        optimizer_param = Nadam(lr=lr_param)
    if optimizer_param == "rmsprop":
        optimizer_param = RMSprop(lr=lr_param)
    if optimizer_param == "sgd":
        optimizer_param = SGD(lr=lr_param)   
    return optimizer_param

def create_dirs():
    list_dir = ["cnn_sparse", "linear_one_hot", "linear_sparse" , "linearX", "nn_sparse", "saves"]
    for dir in list_dir:
        directory = "../models/" + dir 
        if not os.path.exists(directory):
            os.mkdir(directory)