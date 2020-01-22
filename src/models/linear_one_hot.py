import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np
import os
from src.utils.tools import unpickle, get_label_names, display_batch_stat, load_linear_model, y_one_hot, get_optimizer
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD
# from config import isGray

def linear_model(size, nb_output, activation, optimizer, loss,  lr):
    optimizer_param = get_optimizer(optimizer, lr)
    model = tf.keras.Sequential()
    model.add(Dense(1, activation=activation, input_dim=size))
    model.add(Dense(nb_output, activation="softmax", input_dim=1))
    model.compile(optimizer=optimizer_param, loss=loss, metrics=['accuracy'])
    return model

def linear_model_fit(model, X_param, Y_param, epochs, batch_size,  save_path, save_dir, basePath):
    log_dir = basePath + save_dir
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(X_param, Y_param, batch_size=batch_size, verbose=1, epochs=epochs, callbacks=[tensorboard_callback], validation_split=0.2)
    model.save(save_path)
    return model

def predict_linear(model, X):
    img = X.reshape(1, 3072)
    res = np.argmax((model.predict(img)))
    return res

def linear_one_hot(X_all, Y, isTrain,  activation, optimizer, loss, epochs, batch_size, lr, isGray, save_dir):
    if isGray:
        image_size = 32 * 32
    else:
        image_size = 32 * 32 * 3
    Y_one_hot = y_one_hot(Y, max(Y) + 1)
    nb_output = np.shape(Y_one_hot)[1]
    directory = "../models/linear_one_hot/" + save_dir
    if not os.path.exists(directory):
        os.mkdir(directory)
    path = directory + "/model.h5"
    if isTrain:
        model = linear_model(image_size, nb_output,
                             activation, optimizer, loss, lr)

        model = linear_model_fit(model, X_all,Y_one_hot,
                    epochs, batch_size,
                    path, save_dir,  "..\\models\\linear_one_hot\\")
    else:
        model = load_linear_model(path)