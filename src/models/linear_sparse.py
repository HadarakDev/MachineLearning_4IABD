import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np
import os
from src.utils.tools import unpickle, load_linear_model, get_optimizer
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD
from src.utils.models import model_fit

"""This function is create a sparse Linear model

- **parameters**, **types**, **return** and **return types**::
     :param size: number of input
     :param nb_ouptut: number of output ( classes )
     :param activation: activation function
     :param optimizer: optimizer used to train model
     :param loss: loss function to optimize
     :param lr: learning rate
"""


def linear_model(size, nb_output, activation, optimizer, loss, lr):
    optimizer_param = get_optimizer(optimizer, lr)
    model = tf.keras.Sequential()
    model.add(Dense(1, activation=activation, input_dim=size))
    model.add(Dense(nb_output, activation="softmax", input_dim=1))
    model.compile(optimizer=optimizer, loss=loss, metrics=['sparse_categorical_accuracy'])
    return model


def predict_linear(model, X):
    img = X.reshape(1, 3072)
    res = np.argmax((model.predict(img)))
    return res


def linear_sparse(X_all, Y, isTrain, activation, optimizer, loss, epochs, batch_size, lr, isGray, save_dir, base_path):
    if isGray:
        image_size = 32 * 32
    else:
        image_size = 32 * 32 * 3
    nb_output = np.max(Y) + 1
    directory = base_path + save_dir
    if not os.path.exists(directory):
        os.mkdir(directory)
    path = directory + "/model.h5"
    if isTrain:
        model = linear_model(image_size, nb_output,
                             activation, optimizer, loss, lr)

        model = model_fit(model, X_all, Y,
                          epochs, batch_size,
                          path, save_dir, base_path)
    else:
        model = load_linear_model(path)
