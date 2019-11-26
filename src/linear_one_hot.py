import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np
from tools import unpickle, get_label_names, display_batch_stat, load_linear_model, y_one_hot


def linear_model(size, nb_output):
    model = tf.keras.Sequential()
    model.add(Dense(1, activation="linear", input_dim=size))
    model.add(Dense(nb_output, activation="softmax", input_dim=1))
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def linear_model_fit(model, X_param, Y_param, batch_size_param, epochs_param, save_path):
    model.fit(X_param, Y_param, batch_size=batch_size_param, verbose=0, epochs=epochs_param)
    model.save(save_path)
    return model

def predict_liear(model, X):
    img = X.reshape(1, 3072)
    res = np.argmax((model.predict(img)))
    return res

def linear_one_hot(X_all, Y, features, labels, label_names, isTrain, datasetPath):
    Y_one_hot = y_one_hot(Y, max(Y) + 1)
    nb_output = np.shape(Y_one_hot)[1]
    image_size = 32 * 32 * 3
    path = "../models/linear_one_hot/model.h5"
    if isTrain:
        model = linear_model(
                    image_size,
                    nb_output)
        model = linear_model_fit(model, X_all,
                    Y_one_hot,
                    10000,
                    10000,
                    path)
    else:
        model = load_linear_model(path)
    
    for i in range(1000):
        res = predict_liear(model, X_all[i])
        print("predict => " + label_names[res] + " | expected => "  + label_names[Y[i]])