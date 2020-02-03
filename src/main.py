import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from models.cnn_sparse import cnn_sparse
from models.nn_sparse import nn_sparse
from src.models.linear_one_hot import linear_one_hot
from src.models.linear_sparse import linear_sparse
from src.models.unet_conv2D import unet_conv2D
# from src.models.nn_sparse import nn_sparse
# from src.models.nn_one_hot import nn_one_hot
# from src.models.cnn_sparse import cnn_sparse
from src.utils.tools import unpickle, create_dirs,  export_tensorboard, generate_name, display_config


# # Gray
# # X_all = np.mean(X_all.reshape(-1,size * size, 3), axis=2)
#
# Y = np.concatenate(Y)
from utils.data import load_dataset
from utils.export import export_tensorboard_to_csv


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

        # A voir si load norm et non norm de base pour perf
        if isNorm == True:
            X_Final = X_all / 255.0
        else:
            X_Final = X_all

        save_dir = generate_name(data.iloc[i]) + "_sparse"

        array_layers = data['layers'].iloc[i].split("-")
        display_config(activation, optimizer, loss, epochs, batch_size, lr, isGray, isNorm, array_layers, dropout, l1, l2)

        nn_sparse(X_Final, Y, True, activation, optimizer, loss, epochs, batch_size, lr, isGray, save_dir, save_path, array_layers, dropout, l1, l2)

# test(X_all, Y)
def cnn():
    data = pd.read_csv("../config/cnn.csv")
    for i in range(data.shape[0]):
        activation_param = data['activation_param'].iloc[i].split(";")
        optimizer_param = data['optimizer_param'].iloc[i]
        pooling_param = data['pooling_param'].iloc[i]
        lr_param = data['learning_rate'].iloc[i]
        loss_param = "sparse_" + data['loss_param'].iloc[i]
        batch_size_param = data['batch_size_param'].iloc[i]
        epochs_param = data['epochs_param'].iloc[i]
        save_path_info = data['save_path_info'].iloc[i]
        array_layers = data['array_layers'].iloc[i].split(";")
        kernel_shape_param = data['kernel_shape_param'].iloc[i]

        print("START NEW TRAINING")
        print("activation_param : " + str(activation_param))
        print("optimizer_param : " + str(optimizer_param))
        print("learning_rate : " + str(lr_param))
        print("loss_param : " + str(loss_param))
        print("batch_size_param : " + str(batch_size_param))
        print("epochs_param : " + str(epochs_param))
        print("save_path_info : " + str(save_path_info))
        print("array_layers : " + str(array_layers))
        print("kernel_shape_param : " + str(kernel_shape_param))
        print("pooling_param : " + str(pooling_param))


        #launch cnn sparse
        print(X_all.shape)
        cnn_sparse(X_all, Y, True, activation_param, optimizer_param, lr_param, loss_param, batch_size_param,
                   epochs_param, save_path_info, array_layers, pooling_param, kernel_shape_param)
        # save_path_info_sparse = save_path_info.split("_")
        # save_path_info_sparse.insert(3, "sparse")
        # save_path_info_sparse = "_".join(save_path_info_sparse)



if __name__ == "__main__":
    X_all, Y = load_dataset()
    #create_dirs()

    #linear(X_all, Y, "../config/archive/Linear/learning_rate_change2.csv", "..\\models\\Linear\\linear_final\\learning_rate_change\\")
    # export_tensorboard_to_csv("../config/archive/Nn/10_best_more_neurons_test.csv", "../results/Nn/10_best_more_neurons_test.csv",\
    #                          "..\\models\\Nn\\nn_final\\10_best_more_neurons_test\\")

    nn(X_all, Y, "../config/archive/Nn/5_top_with_regularizers.csv", "..\\models\\Nn\\nn_final\\5_top_with_regularizers\\")
    #cnn()
    #export_tensorboard()
    #renameWithNorm()
    # test(X_all, Y)
    #renameSyntax()
    #tanh,adagrad,categorical_crossentropy,100,5000,0.0001,32-32-32-32,False,True,0,0,0
    # unet_conv2D(X_all, Y, True, "tanh", "adagrad", 0.001, "sparse_categorical_crossentropy", 5000,
    #             1000, "my_cool_unet", [64, 64], 2)