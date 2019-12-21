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

def create_dic(f_name, loss, acc, is_one_hot, is_norm, is_gray):
    return {'Name': f_name,
        'activation': [elt.split("_")[0] for elt in f_name],
        'optimizer': [elt.split("_")[1] for elt in f_name],
        'is_sparse': is_one_hot,
        'is_norm': is_norm,
        'is_gray': is_gray,
        'loss': loss,
        'accuracy': acc}


def export_tensorboard():
    model_type_dir = os.listdir("../models")
    f_name = []
    loss = []
    acc = []
    for d in model_type_dir:
        if os.path.isdir("../models/" + d):
            model_dir = os.listdir("../models/" + d)
            for md in model_dir:
                #print("file: " + md)
                file_name = "../models/" + d + "/" + md + "/train/"
                ea = event_accumulator.EventAccumulator(path=file_name)
                ea.Reload()
                loss.append(ea.Scalars('epoch_loss')[-1][2])
                f_name.append(md)
                try:
                    acc.append(ea.Scalars('epoch_accuracy')[-1][2])
                except:
                    acc.append(ea.Scalars('epoch_sparse_categorical_accuracy')[-1][2])
        
    # Split names to get only optimizer + activation
    f_names_short = []
    for f in f_name:
        f_names_short.append("_".join(f.split("_", 2)[:2]))

    # generate dataframes with values
    frames = []
    sparse = [True if 'sparse' in x else False for x in f_name]
    gray = [True if 'gray' in x else False for x in f_name]
    norm = [True if 'norm' in x else False for x in f_name]

    frames.append(pd.DataFrame(create_dic(f_names_short, loss, acc, sparse, gray, norm)))

    # Merge all
    result = pd.concat(frames)
    print(result)
    # Save to csv
    result.to_csv("../one_hot_vs_sparse.csv", index=False)

export_tensorboard()
