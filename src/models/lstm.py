import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Model
import numpy as np
import os

from tensorflow_core.python.keras.layers import Flatten
from tensorflow_core.python.keras.regularizers import L1L2

from utils.tools import unpickle, load_linear_model, get_optimizer
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD
from utils.models import model_fit

import matplotlib.pyplot as plt
from tensorflow.keras.activations import *
from tensorflow.keras.datasets import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model

# class PrintTrueTrainMetricsAtEpochEnd(Callback):
#     def __init__(self, x_train, y_train):
#         super().__init__()
#         self.x_train = x_train
#         self.y_train = y_train

#     def on_epoch_end(self, epoch, logs=None):
#         loss, acc = self.model.evaluate(self.x_train, self.y_train, batch_size=8192)
#         print(f"Le Vrai loss du train : {loss}")
#         print(f"La Vrai acc du train : {acc}")

def lstm_model(image_size, nb_output, activation, optimizer, loss, lr, array_layers, dropout, l1, l2):
    model = Sequential()
    #input_tensor = Input(32, 32)
    optimizer_param = get_optimizer(optimizer, lr)
    model.add(LSTM(int(array_layers[0]), return_sequences=True))
    for i in range(1, len(array_layers) - 2):
        model.add(LSTM(int(array_layers[i]), return_sequences=True))
    model.add(LSTM(int(array_layers[len(array_layers)-1]), return_sequences=False))
    model.add(Dense(10, activation=activation))
    #model = Model(input_tensor, dense_tensor)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[sparse_categorical_accuracy])
    return model


def lstm_sparse(X_all, Y, isTrain, activation, optimizer, loss, epochs, batch_size, lr, isGray, save_dir, base_path,
              array_layers, dropout, l1, l2):
    print(X_all)
    if isGray:
        X_all = np.mean(X_all.reshape(-1, 32 * 32, 3), axis=2)
        image_size = 32 * 32
    else:
        image_size = 32 * 32 * 3
    nb_output = np.max(Y) + 1
    directory = base_path + save_dir
    print(directory)
    if not os.path.exists(directory):
        os.mkdir(directory)
    path = directory + "/model.h5"
    #X=X_all
    X = np.reshape(X_all, (X_all.shape[0], 1, X_all.shape[1]))
    if isTrain:
        model = lstm_model(image_size, nb_output, activation, optimizer, loss, lr, array_layers, dropout, l1, l2) # create LSTM

        model = model_fit(model, X, Y,
                          epochs, batch_size,
                          path, save_dir, base_path)