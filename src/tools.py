import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from tensorboard.backend.event_processing import event_accumulator
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

def export_tensorboard():
    model_type_dir = os.listdir("../models")
    acc_sparse = []
    acc = []
    loss_sparse = []
    loss = []
    f_name_sparse = []
    f_name = []
    for d in model_type_dir:
        if os.path.isdir("../models/" + d):
            model_dir = os.listdir("../models/" + d)
            for md in model_dir:
                #print("file: " + md)
                path_file = "../models/" + d + "/" + md + "/train/"
                ea = event_accumulator.EventAccumulator(path=path_file)
                ea.Reload()
                if "sparse" in path_file: 
                    acc_sparse.append(ea.Scalars('epoch_sparse_categorical_accuracy')[-1][2])
                    loss_sparse.append(ea.Scalars('epoch_loss')[-1][2])
                    f_name_sparse.append(md)
                else:
                    acc.append(ea.Scalars('epoch_accuracy')[-1][2])
                    loss.append(ea.Scalars('epoch_loss')[-1][2])
                    f_name.append(md)

    print(acc_sparse)
    print(loss_sparse)
    print(acc)
    print(loss)
    for i, f in enumerate(f_name):
        f_name[i] = "_".join(f.split("_", 2)[:2])
    for i, f in enumerate(f_name_sparse):
        f_name_sparse[i] = "_".join(f.split("_", 2)[:2])
    print(f_name)
    print(f_name_sparse)
    dict = {'Name': f_name_sparse,
        'activation': [elt.split("_")[0] for elt in f_name_sparse],
        'optimizer': [elt.split("_")[1] for elt in f_name_sparse],
        'is_one_hot': [True] * len(f_name_sparse),
        'loss': loss_sparse,
        'accuracy': acc_sparse}          
    df_sparse = pd.DataFrame(dict) 
    print(df_sparse)
    dict2 = {'Name': f_name,
        'activation': [elt.split("_")[0] for elt in f_name],
        'optimizer': [elt.split("_")[1] for elt in f_name],
        'is_one_hot': [False] * len(f_name),
        'loss': loss,
        'accuracy': acc}
    df = pd.DataFrame(dict2)
    frames = [df_sparse, df]
    result = pd.concat(frames)
    print(result)
    result.to_csv("../one_hot_vs_sparse.csv", index=False)

export_tensorboard()
