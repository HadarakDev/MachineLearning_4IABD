import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np
import os
from tools import unpickle, get_label_names, display_batch_stat, load_linear_model


def linear_model(size, nb_output, activation_param, optimizer_param, loss_param):
    model = tf.keras.Sequential()
    model.add(Dense(1, activation=activation_param, input_dim=size))
    model.add(Dense(nb_output, activation="softmax", input_dim=1))
    model.compile(optimizer=optimizer_param, loss=loss_param, metrics=['sparse_categorical_accuracy'])
    return model

def linear_model_fit(model, X_param, Y_param, batch_size_param, epochs_param, save_path, save_path_info):
    log_dir = "..\\models\\linear_sparse\\" + save_path_info
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(X_param, Y_param, batch_size=batch_size_param, verbose=1, epochs=epochs_param, callbacks=[tensorboard_callback], validation_split=0.2)
    model.save(save_path)
    return model

def predict_linear(model, X):
    img = X.reshape(1, 3072)
    res = np.argmax((model.predict(img)))
    return res

def linear_sparse(X_all, Y, isTrain,  activation_param, optimizer_param, loss_param, batch_size_param, epochs_param, save_path_info):
    nb_output = np.max(Y) + 1
    image_size = 32 * 32 * 3
    directory = "../models/linear_sparse/" + save_path_info
    if not os.path.exists(directory):
        os.mkdir(directory)
    path = directory + "/" + save_path_info + ".h5"
    if isTrain:
        model = linear_model(
                    image_size,
                    nb_output,
                    activation_param,
                    optimizer_param,
                    loss_param)
        model = linear_model_fit(model, X_all,
                    Y,
                    batch_size_param,
                    epochs_param,
                    path, save_path_info)
    else:
        model = load_linear_model(path)