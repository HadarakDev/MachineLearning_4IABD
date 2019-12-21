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
        'is_one_hot': [is_one_hot] * len(f_name),
        'is_norm': [is_norm] * len(f_name),
        'is_gray': [is_gray] * len(f_name),
        'loss': loss,
        'accuracy': acc}


def export_tensorboard():
    model_type_dir = os.listdir("../models")
    acc = [[],[],[],[],[],[],[],[]]
    loss = [[],[],[],[],[],[],[],[]]
    f_name = [[],[],[],[],[],[],[],[]]
    for d in model_type_dir:
        if os.path.isdir("../models/" + d):
            model_dir = os.listdir("../models/" + d)
            for md in model_dir:
                #print("file: " + md)
                path_file = "../models/" + d + "/" + md + "/train/"
                ea = event_accumulator.EventAccumulator(path=path_file)
                ea.Reload()
                # Sparse only
                if "sparse" in path_file and "norm" not in path_file and "gray" not in path_file:
                    i = 0
                    acc[i].append(ea.Scalars('epoch_sparse_categorical_accuracy')[-1][2])
                    loss[i].append(ea.Scalars('epoch_loss')[-1][2])
                    f_name[i].append(md)
                    continue
                # Nothing
                if "sparse" not in path_file and "norm" not in path_file and "gray" not in path_file:
                    i = 1
                    acc[i].append(ea.Scalars('epoch_sparse_categorical_accuracy')[-1][2])
                    loss[i].append(ea.Scalars('epoch_loss')[-1][2])
                    f_name[i].append(md)
                    continue
                # Sparse + norm
                if "sparse" in path_file and "norm" in path_file and "gray" not in path_file:
                    i = 2
                    acc[i].append(ea.Scalars('epoch_sparse_categorical_accuracy')[-1][2])
                    loss[i].append(ea.Scalars('epoch_loss')[-1][2])
                    f_name[i].append(md)
                    continue
                # Norm
                if "sparse" not in path_file and "norm" in path_file and "gray" not in path_file:
                    i = 3
                    acc[i].append(ea.Scalars('epoch_sparse_categorical_accuracy')[-1][2])
                    loss[i].append(ea.Scalars('epoch_loss')[-1][2])
                    f_name[i].append(md)
                    continue
                # Sparse Gray
                if "sparse" in path_file and "norm" not in path_file and "gray" in path_file:
                    i = 4
                    acc[i].append(ea.Scalars('epoch_sparse_categorical_accuracy')[-1][2])
                    loss[i].append(ea.Scalars('epoch_loss')[-1][2])
                    f_name[i].append(md)
                    continue
                # Gray
                if "sparse" not in path_file and "norm" not in path_file and "gray" in path_file:
                    i = 5
                    acc[i].append(ea.Scalars('epoch_sparse_categorical_accuracy')[-1][2])
                    loss[i].append(ea.Scalars('epoch_loss')[-1][2])
                    f_name[i].append(md)
                    continue
                # Sparse + Gray + Norm
                if "sparse" in path_file and "norm" in path_file and "gray" in path_file:
                    i = 6
                    acc[i].append(ea.Scalars('epoch_sparse_categorical_accuracy')[-1][2])
                    loss[i].append(ea.Scalars('epoch_loss')[-1][2])
                    f_name[i].append(md)
                    continue
                # Gray + Norm
                if "sparse" not in path_file and "norm" in path_file and "gray" in path_file:
                    i = 7
                    acc[i].append(ea.Scalars('epoch_sparse_categorical_accuracy')[-1][2])
                    loss[i].append(ea.Scalars('epoch_loss')[-1][2])
                    f_name[i].append(md)
                    continue
            


    # Split names to get only
    # for ici
    for i in range(8):
        for j, f in enumerate(f_name[0]):
            f_name[i][j] = "_".join(f.split("_", 2)[:2])

    # generate dataframes with values
    frames = []
    # Sparse only
    frames.append(pd.DataFrame(create_dic(f_name[0], loss[0], acc[0], True, False, False)))
    # Nothing
    frames.append(pd.DataFrame(create_dic(f_name[1], loss[1], acc[1], False, False, False)))
    # Sparse + norm
    frames.append(pd.DataFrame(create_dic(f_name[2], loss[2], acc[2], True, True, False)))
    # Norm
    frames.append(pd.DataFrame(create_dic(f_name[3], loss[3], acc[3], False, True, False)))
    # Sparse Gray
    frames.append(pd.DataFrame(create_dic(f_name[4], loss[4], acc[4], True, False, True)))
    # Gray
    frames.append(pd.DataFrame(create_dic(f_name[5], loss[5], acc[5], False, False, True)))
    # Sparse + Gray + Norm
    frames.append(pd.DataFrame(create_dic(f_name[6], loss[6], acc[6], True, True, True)))
    # Gray + Norm
    frames.append(pd.DataFrame(create_dic(f_name[7], loss[7], acc[7], False, True, True)))

    # Merge all
    result = pd.concat(frames)
    print(result)
    # Save to csv
    result.to_csv("../one_hot_vs_sparse.csv", index=False)

#export_tensorboard()
