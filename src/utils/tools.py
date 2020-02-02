import pickle
from os import path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from tensorboard.backend.event_processing import event_accumulator
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

def display_config(activation, optimizer, loss, epochs, batch_size, lr, isGray, isNorm, layers = None):
    print("START NEW TRAINING")
    print("activation : " + str(activation))
    print("optimizer : " + str(optimizer))
    print("loss : " + str(loss))

    print("epochs : " + str(epochs))
    print("batch-size : " + str(batch_size))
    print("learning-rate : " + str(lr))

    print("isGray : " + str(isGray))
    print("isNorm : " + str(isNorm))

    if layers != None:
        print("array_layers : " + str(layers))


def generate_name(config):
    name = ""
    for k in config:
        name += str(k) + "_"
    name = name[0:-1]
    return name



def unpickle(path_batch, size, isRGB):
    with open(path_batch, 'rb') as fo:
        batch = pickle.load(fo, encoding='latin1')
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    # Make a resize here depending on 'size' and 'isRBG'
    labels = batch['labels']
    return features, labels


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
    list_dir = ["cnn_sparse", "linear_one_hot", "linear_sparse" , "linearX", "nn_sparse", "nn_one_hot", "cnn_sparse", "sparse_vs_oneHot_Linear"]
    for dir in list_dir:
        directory = "../models/" + dir 
        if not os.path.exists(directory):
            os.mkdir(directory)

def create_dic(f_name, loss, acc, is_one_hot, is_norm, is_gray, loss_val, acc_val, drop, l1, l2):
    return {'Name': f_name,
        'activation': [elt.split("_")[0] for elt in f_name],
        'optimizer': [elt.split("_")[1] for elt in f_name],
        'is_sparse': is_one_hot,
        'is_norm': is_norm,
        'is_gray': is_gray,
        'loss': loss,
        'accuracy': acc,
        'dropout': drop,
        'l1': l1,
        'l2': l2,
        'loss_validation': loss_val,
        'accuracy_validation': acc_val}


def export_tensorboard():
    model_type_dir = os.listdir("../models")
    f_name = []
    loss = []
    loss_val= []
    acc = []
    acc_val = []
    for d in model_type_dir:
        if os.path.isdir("../models/" + d):
            if d == "nn_one_hot" or d == "nn_sparse": ######## CHANGE ME => FOLDER SELECTION
                model_dir = os.listdir("../models/" + d)
                for md in model_dir:
                    print("file: " + md)
                    file_name = "../models/" + d + "/" + md + "/train/"
                    file_name_validation = "../models/" + d + "/" + md + "/validation/"
                    ea = event_accumulator.EventAccumulator(path=file_name)
                    ea_val = event_accumulator.EventAccumulator(path=file_name_validation)
                    ea.Reload()
                    ea_val.Reload()
                    loss.append(ea.Scalars('epoch_loss')[-1][2])
                    loss_val.append(ea_val.Scalars('epoch_loss')[-1][2])
                    f_name.append(md)
                    try:
                        acc.append(ea.Scalars('epoch_accuracy')[-1][2])
                        acc_val.append(ea_val.Scalars('epoch_accuracy')[-1][2])
                    except:
                        acc.append(ea.Scalars('epoch_sparse_categorical_accuracy')[-1][2])
                        acc_val.append(ea_val.Scalars('epoch_sparse_categorical_accuracy')[-1][2])
        
    # Split names to get only optimizer + activation
    f_names_short = []
    print(f_name)
    for f in f_name:
        f_names_short.append("_".join(f.split("_", 2)[:2]))

    # generate dataframes with values
    frames = []
    sparse = [True if 'sparse' in x else False for x in f_name]
    gray = [True if 'gray' in x else False for x in f_name]
    norm = [True if 'norm' in x else False for x in f_name]
    drop = [float(x[x.index("dropout") + 7:x.index("dropout") + 9])/10 if "dropout" in x else 0 for x in f_name]
    l1 = [True if 'l1' in x else False for x in f_name]
    l2 = [True if 'l2' in x else False for x in f_name]

    frames.append(pd.DataFrame(create_dic(f_name, loss, acc, sparse, norm, gray, loss_val, acc_val, drop, l1, l2)))

    # Merge all
    result = pd.concat(frames)
    print(result)
    # Save to csv
    result.to_csv("../results_NN.csv", index=False)