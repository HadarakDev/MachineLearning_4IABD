import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow_core.python.keras.backend import clear_session

from models.cnn_sparse import cnn_sparse
from models.nn_sparse import nn_sparse
from models.resnet import resnet_model_dense
from models.unet_conv2D import unet_conv2D_sparse
from src.models.linear_one_hot import linear_one_hot
from src.models.linear_sparse import linear_sparse
# from src.models.nn_sparse import nn_sparse
# from src.models.nn_one_hot import nn_one_hot
# from src.models.cnn_sparse import cnn_sparse
from src.utils.tools import unpickle, create_dirs, export_tensorboard, generate_name, display_config, \
    display_weights_number
from tensorflow.keras import Model


# # Gray
# # X_all = np.mean(X_all.reshape(-1,size * size, 3), axis=2)
#
# Y = np.concatenate(Y)
from utils.data import load_dataset
from utils.export import export_tensorboard_to_csv, export_tensorboard_regularizers


def compareOneHotAndSparse(X_all, Y):
    data = pd.read_csv("../config/sparse_vs_oneHotLinear.csv")
    for i in range(data.shape[0]):

        activation = data['activation'].iloc[i]
        optimizer = data['optimizer'].iloc[i]
        loss = data['loss'].iloc[i]

        epochs = data['epochs'].iloc[i]
        batch_size = data['batch-size'].iloc[i]
        lr = data['learning-rate'].iloc[i]

        isGray = data['isGray'].iloc[i]
        isNorm = data['isNorm'].iloc[i]

        loss_sparse = "sparse_" + loss
        display_config(activation, optimizer, loss, epochs, batch_size, lr, isGray, isNorm)
        if isNorm == True:
            X_Final = X_all / 255.0
        else:
            X_Final = X_all

        save_dir = generate_name(data.iloc[i])
        linear_one_hot(X_Final, Y, True, activation, optimizer, loss, epochs, batch_size, lr, isGray, save_dir,  "..\\models\\sparse_vs_oneHot_Linear\\")

        #save_dir_sparse = save_dir + "_sparse"
        linear_sparse(X_Final, Y, True, activation, optimizer, loss_sparse, epochs, batch_size, lr, isGray, save_dir,  "..\\models\\sparse_vs_oneHot_Linear\\")


def linear(X_all, Y, config_path, save_path):
    data = pd.read_csv(config_path)
    for i in range(data.shape[0]):

        activation = data['activation'].iloc[i]
        optimizer = data['optimizer'].iloc[i]
        loss = "sparse_" + data['loss'].iloc[i]

        epochs = data['epochs'].iloc[i]
        batch_size = data['batch-size'].iloc[i]
        lr = data['learning-rate'].iloc[i]

        isGray = data['isGray'].iloc[i]
        isNorm = data['isNorm'].iloc[i]

        display_config(activation, optimizer, loss, epochs, batch_size, lr, isGray, isNorm)

        #A voir si load norm et non norm de base pour perf
        if isNorm == True:
            X_Final = X_all / 255.0
        else:
            X_Final = X_all

        save_dir = generate_name(data.iloc[i]) + "_sparse"
        linear_sparse(X_Final, Y, True, activation, optimizer, loss, epochs, batch_size, lr, isGray, save_dir, save_path)

def nn(X_all, Y, config_path, save_path):
    data = pd.read_csv(config_path)
    for i in range(data.shape[0]):

        activation = data['activation'].iloc[i]
        optimizer = data['optimizer'].iloc[i]
        loss = "sparse_" + data['loss'].iloc[i]

        epochs = data['epochs'].iloc[i]
        batch_size = data['batch-size'].iloc[i]
        lr = data['learning-rate'].iloc[i]

        isGray = data['isGray'].iloc[i]
        isNorm = data['isNorm'].iloc[i]

        dropout = data["Dropout"].iloc[i]
        l1 = data["L1"].iloc[i]
        l2 = data["L2"].iloc[i]

        if isNorm == True:
            X_Final = X_all / 255.0
        else:
            X_Final = X_all

        save_dir = generate_name(data.iloc[i]) + "_sparse"

        array_layers = data['layers'].iloc[i].split("-")
        display_config(activation, optimizer, loss, epochs, batch_size, lr, isGray, isNorm, array_layers, dropout, l1, l2)

        nn_sparse(X_Final, Y, True, activation, optimizer, loss, epochs, batch_size, lr, isGray, save_dir, save_path, array_layers, dropout, l1, l2)

# test(X_all, Y)
def cnn(X_all, Y, config_path, save_path):
    data = pd.read_csv(config_path)
    for i in range(data.shape[0]):
        if i % 25 == 0:
            clear_session()
        activation = data['activation'].iloc[i]
        optimizer = data['optimizer'].iloc[i]
        loss = "sparse_" + data['loss'].iloc[i]

        epochs = data['epochs'].iloc[i]
        batch_size = data['batch-size'].iloc[i]
        lr = data['learning-rate'].iloc[i]

        isGray = data['isGray'].iloc[i]
        isNorm = data['isNorm'].iloc[i]

        dropout = data["Dropout"].iloc[i]
        l1 = data["L1"].iloc[i]
        l2 = data["L2"].iloc[i]
        pooling = data["pooling"].iloc[i]
        kernel_shape = data["kernel-shape"].iloc[i]

        if isNorm == True:
            X_Final = X_all / 255.0
        else:
            X_Final = X_all

        save_dir = generate_name(data.iloc[i]) + "_sparse"

        array_layers = data['layers'].iloc[i].split("-")
        display_config(activation, optimizer, loss, epochs, batch_size, lr, isGray, isNorm, array_layers, dropout, l1, l2)
        try:
            cnn_sparse(X_Final, Y, True, activation, optimizer, loss, epochs, batch_size, lr, isGray, save_dir, save_path,
                  array_layers, pooling, kernel_shape,  dropout, l1, l2)
        except ValueError:
            clear_session()
            print("fail")





def unet_conv2d(X_all, Y, config_path, save_path):
    data = pd.read_csv(config_path)
    for i in range(data.shape[0]):

        if i % 25 == 0:
            clear_session()
        activation = data['activation'].iloc[i]
        optimizer = data['optimizer'].iloc[i]
        loss = "sparse_" + data['loss'].iloc[i]

        epochs = data['epochs'].iloc[i]
        batch_size = data['batch-size'].iloc[i]
        lr = data['learning-rate'].iloc[i]

        isGray = data['isGray'].iloc[i]
        isNorm = data['isNorm'].iloc[i]

        dropout = data["Dropout"].iloc[i]
        l1 = data["L1"].iloc[i]
        l2 = data["L2"].iloc[i]
        pooling = data["pooling"].iloc[i]
        kernel_shape = data["kernel-shape"].iloc[i]

        if isNorm == True:
            X_Final = X_all / 255.0
        else:
            X_Final = X_all

        save_dir = generate_name(data.iloc[i]) + "_sparse"

        array_layers = data['layers'].iloc[i].split("-")
        array_layers = [int(x) for x in array_layers]
        display_config(activation, optimizer, loss, epochs, batch_size, lr, isGray, isNorm, array_layers, dropout, l1, l2)

        try:
            unet_conv2D_sparse(X_Final, Y, True, activation, optimizer, loss, epochs, batch_size, lr, isGray, save_dir, save_path, array_layers, pooling, kernel_shape,  dropout, l1, l2)
        except ValueError:
            clear_session()
            print("fail")

def resnet_dense(X_all, Y, config_path, save_path):
    data = pd.read_csv(config_path)
    for i in range(data.shape[0]):

        activation = data['activation'].iloc[i]
        optimizer = data['optimizer'].iloc[i]
        loss = "sparse_" + data['loss'].iloc[i]

        epochs = data['epochs'].iloc[i]
        batch_size = data['batch-size'].iloc[i]
        lr = data['learning-rate'].iloc[i]

        isGray = data['isGray'].iloc[i]
        isNorm = data['isNorm'].iloc[i]

        dropout = data["Dropout"].iloc[i]
        l1 = data["L1"].iloc[i]
        l2 = data["L2"].iloc[i]

        if isNorm == True:
            X_Final = X_all / 255.0
        else:
            X_Final = X_all

        save_dir = generate_name(data.iloc[i]) + "_sparse"

        array_layers = data['layers'].iloc[i].split("-")
        display_config(activation, optimizer, loss, epochs, batch_size, lr, isGray, isNorm, array_layers, dropout, l1, l2)

        resnet_model_dense(X_Final, Y, True, activation, optimizer, loss, epochs, batch_size, lr, isGray, save_dir, save_path,
                  array_layers, 2, dropout, l1, l2)

if __name__ == "__main__":
    X_all, Y = load_dataset()
    #create_dirs()
    #display_weights_number("../config/archive/Unet/optimizer_activaction_testing.csv", "..\\models\\Unet\\optimizer_activaction_testing\\")
    #linear(X_all, Y, "../config/archive/Linear/learning_rate_change2.csv", "..\\models\\Linear\\linear_final\\learning_rate_change\\")
    #export_tensorboard_to_csv("../config/archive/Unet/5_top_regularizers.csv", "../results/Unet/5_top_regularizers.csv", "..\\models\\Unet\\5_top_regularizers\\")
    # export_tensorboard_to_csv("../config/archive/Nn/5_top_with_regularizers_2.csv",
    #                           "../results/Nn/5_top_with_regularizers_Dropout_execpt_first.csv",
    #                           "..\\models\\Nn\\nn_final\\5_top_with_regularizers_2\\")
    #X = X_all.reshape(50000, 32, 32, 3)
    #export_tensorboard_regularizers("../config/archive/Cnn/5_top_with_regularizers_2.csv", "../results/Cnn/5_top_with_regularizers_final.csv", "..\\models\\Cnn\\cnn_final\\5_top_with_regularizers_2\\",  X_all, Y)

    # Coder function export avec dropout
    #print(os.path.exists("..\\models\\Nn\\nn_final\\5_top_with_regularizers\\elu_adam_categorical_crossentropy_500_1000_0.0001_1024-1024-1024-1024_False_True_0.2_0.0_0.01_sparse\\model.h5"))
    #nn(X_all, Y, "../config/archive/Nn/5_top_with_regularizers_2.csv", "..\\models\\Nn\\nn_final\\5_top_with_regularizers_2\\")
    #cnn(X_all, Y, "../config/archive/Cnn/5_top_with_regularizers_2.csv", "..\\models\\Cnn\\cnn_final\\5_top_with_regularizers_2\\")
    #clear_session()
    #unet_conv2d(X_all, Y, "../config/archive/Unet/5_top_regularizers.csv", "..\\models\\Unet\\5_top_regularizers\\")
    #resnet_dense(X_all, Y, "../config/archive/ResNet/explo.csv", "..\\models\\ResNet\\explo\\")
    # test(X_all, Y)
    #renameSyntax()


    # unet_conv2D(X_all, Y, True, "tanh", "adagrad", 0.001, "sparse_categorical_crossentropy", 5000,
    #             1000, "my_cool_unet", [64, 64], 2)