import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os

from tensorflow_core.python.keras.layers import MaxPool2D, AveragePooling2D

from src.utils.tools import unpickle, load_linear_model, get_optimizer
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD

from src.utils.models import model_fit


def cnn_model(nb_output, activation_param, optimizer_param, lr_param, loss_param, array_layers, pooling_param, kernel_shape_param):
    optimizer_param = get_optimizer(optimizer_param, lr_param)
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=array_layers[0], kernel_size=(kernel_shape_param, kernel_shape_param), padding='same', activation=activation_param[0], input_shape=(32, 32, 3)))
    if pooling_param == "avg_pool":
        model.add(tf.keras.layers.AveragePooling2D((2, 2), padding='same'))
    else:
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))

    for i in range(1, len(array_layers)):
        model.add(tf.keras.layers.Conv2D(array_layers[i], (kernel_shape_param, kernel_shape_param), padding='same',  activation=activation_param[i]))
        if pooling_param == "avg_pool":
            model.add(tf.keras.layers.AveragePooling2D((2, 2), padding='same'))
        else:
            model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
    model.add(tf.keras.layers.Flatten())
    model.add(Dense(nb_output, activation="softmax"))
    print(loss_param)
    model.compile(optimizer=optimizer_param, loss=loss_param, metrics=['sparse_categorical_accuracy'])
    model.summary()
    return model



def predict_cnn(model, X):
    img = X.reshape(1, 3072)
    res = np.argmax((model.predict(img)))
    return res

def cnn_sparse(X_all, Y, isTrain,  activation_param, optimizer_param, lr_param, loss_param, batch_size_param, epochs_param, save_path_info, array_layers, pooling_param, kernel_shape_param):
    nb_output = np.max(Y) + 1
    image_size = 32 * 32 * 3
    X = X_all.reshape(50000, 32, 32, 3)

    kernel_shape_param = int(kernel_shape_param)
    array_layers = [int(x) for x in array_layers]
    directory = "../models/cnn_sparse/" + save_path_info
    if not os.path.exists(directory):
        os.mkdir(directory)
    path = directory + "/" + save_path_info + ".h5"
    if isTrain:
        model = cnn_model(
                    nb_output,
                    activation_param,
                    optimizer_param,
                    lr_param,
                    loss_param,
                    array_layers,
                    pooling_param,
                    kernel_shape_param)
        model = model_fit(model, X,
                    Y,
                    batch_size_param,
                    epochs_param,
                    path, save_path_info)
    else:
        model = load_linear_model(path)

