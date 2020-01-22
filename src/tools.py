import pickle
from os import path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from tensorboard.backend.event_processing import event_accumulator
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

def display_config(activation, optimizer, loss, epochs, batch_size, lr, isGray, isNorm):
    print("START NEW TRAINING")
    print("activation : " + str(activation))
    print("optimizer : " + str(optimizer))
    print("loss : " + str(loss))

    print("epochs : " + str(epochs))
    print("batch-size : " + str(batch_size))
    print("learning-rate : " + str(lr))

    print("isGray : " + str(isGray))
    print("isNorm : " + str(isNorm))


def generate_name(config):
    name = ""
    for k in config:
        name += str(k) + "_"
    name = name[0:-1]


## add Norm at the end of Dir/Files
def renameWithNorm():
    basePath = "..\\models\\cnn_sparse\\"
    renameDirNorm(basePath)
    renameFilesNorm(basePath)

def renameDirNorm(basePath):
    dirs = os.listdir(basePath)
    for dir in dirs:
        if dir.find("norm") == -1:
            os.rename(basePath + dir, basePath + dir + "_norm")

def renameFilesNorm(basePath):
    dirs = os.listdir(basePath)
    for dir in dirs:
        files = os.listdir(basePath + dir)
        for file in files:
            if file.find("h5") != -1 and file.find("norm") == -1:
                tmpFile = file[0:-3] + "_norm.h5"
                os.rename(basePath + dir + "//" + file, basePath + dir + "//" + tmpFile)



## rename dir/file syntax
def renameSyntax():
    basePath = "..\\models\\nn_sparse\\"

    renameSyntaxDir(basePath)
    renameSyntaxFile(basePath)

def renameSyntaxDir(basePath):
    dirs = os.listdir(basePath)
    for dir in dirs:
        if dir.find("dropout") != -1:
            newDir = dir.replace("dropout_02", "dropout02")
            newDir = newDir.replace("dropout_01", "dropout01")
            if not path.exists(basePath + newDir):
                os.rename(basePath + dir, basePath + newDir)


def renameSyntaxFile(basePath):
    dirs = os.listdir(basePath)
    for dir in dirs:
        files = os.listdir(basePath + dir)
        for file in files:
            if file.find("dropout") != -1:
                newFile = file.replace("dropout_02", "dropout02")
                newFile = newFile.replace("dropout_01", "dropout01")
                if not path.exists(basePath + dir + "//" + newFile):
                    os.rename(basePath + dir + "//" + file, basePath + dir + "//" + newFile)



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
    list_dir = ["cnn_sparse", "linear_one_hot", "linear_sparse" , "linearX", "nn_sparse", "nn_one_hot", "cnn_sparse"]
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