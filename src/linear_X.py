import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
from tools import unpickle, get_label_names, display_batch_stat, load_linear_model
import os


def linear_X_models(size, activation_param, optimizer_param, loss_param):
    model = tf.keras.Sequential()
    model.add(Dense(1, activation=activation_param, input_dim=size))
    model.compile(optimizer=optimizer_param,
                  loss=loss_param, metrics=["accuracy"])
    return model


def linear_X_models_fit(model, X_param, Y_param, batch_size_param, epochs_param, save_path, save_path_info):
    log_dir = "..\\models\\linearX\\" + save_path_info
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(X_param, Y_param, batch_size=batch_size_param,
              verbose=1, epochs=epochs_param,
              callbacks=[tensorboard_callback])
    model.save(save_path)
    return model


def predict_linear(model_all, X):
    res = []
    for model in model_all:
        img = X.reshape(1, 3072)
        res.append(model.predict(img)[0][0])
    print(res)
    return res.index(max(res))


def linear_X(X_all, Y, isTrain, activation_param, optimizer_param, loss_param, batch_size_param, epochs_param, save_path_info):
    image_size = 32 * 32 * 3
    Y_all = []
    model_all = []
    nb_output = max(Y) + 1
    directory = "../models/linearX/" + save_path_info
    if not os.path.exists(directory):
        os.mkdir(directory)
    path = "../models/linearX/" + save_path_info + "/model_" + save_path_info + "_{}.h5"

    for i in range(nb_output):
        Y_all.append(np.array([1 if y == i else 0 for y in Y]))

    if isTrain:
        for i in range(nb_output):
            model_all.append(linear_X_models(
                image_size, activation_param, optimizer_param, loss_param))
        for i in range(nb_output):
            model_all[i] = linear_X_models_fit(model_all[i],
                                               X_all,
                                               Y_all[i],
                                               batch_size_param,
                                               epochs_param,
                                               path.format(str(i)),
                                               save_path_info)
    else:
        for i in range(nb_output):
            model_all.append(load_linear_model(path.format(str(i))))

#    result = []
#    range_img = 1000
#    for i in range(range_img):
#        result.append(predict_linear(model_all, X_all[i]))
