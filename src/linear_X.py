import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np
from tools import unpickle, get_label_names, display_batch_stat, load_linear_model


def linear_X_models(size):
    model = tf.keras.Sequential()
    model.add(Dense(1, activation="sigmoid", input_dim=size))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def linear_X_models_fit(model, X_param, Y_param, batch_size_param, epochs_param, save_path):
    model.fit(X_param, Y_param, batch_size=batch_size_param, verbose=0, epochs=epochs_param)
    model.save(save_path) 
    return model

def predict_linear(model_all, X):
    res = []
    for model in model_all:
        img = X.reshape(1, 3072)
        res.append(model.predict(img)[0][0])
    print(res)
    return res.index(max(res))

def linear_X(X_all, Y, features, labels, label_names, isTrain, datasetPath):
    image_size = 32 * 32 * 3
    Y_all = []
    model_all = []
    nb_output = max(Y) + 1
    path = "../models/linear10models/model{}.h5"

    for i in range(nb_output):
        Y_all.append(np.array([1 if y == i else 0 for y in Y]))

    if isTrain:
        for i in range(nb_output):
            model_all.append(linear_X_models(image_size))
        for i in range(nb_output):
            model_all[i] = linear_X_models_fit(model_all[i],
                                        X_all,
                                        Y_all[i],
                                        10000,
                                        200,
                                        path.format(str(i)))
    else:
        for i in range(nb_output):
            model_all.append(load_linear_model(path.format(str(i)))) 

    result= [] 
    range_img = 1000
    for i in range(range_img):
        result.append(predict_linear(model_all, X_all[i]))
