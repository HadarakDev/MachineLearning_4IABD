import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Model
import numpy as np
import os
from src.utils.tools import unpickle,  load_linear_model, get_optimizer
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD
from src.utils.models import model_fit


def nn_model(image_size, nb_output, activation, optimizer, loss, lr, array_layers):
    optimizer_param = get_optimizer(optimizer, lr)
    model = tf.keras.Sequential()

    model.add(Dense(array_layers[0], activation=activation, input_dim=image_size))

    for i in range(1, len(array_layers)):
        model.add(Dense(array_layers[i], activation=activation))

    model.add(Dense(nb_output, activation="softmax"))
    print(loss)
    model.compile(optimizer=optimizer, loss=loss, metrics=['sparse_categorical_accuracy'])
    model.summary()
    return model

def predict_nn(model, X):
    img = X.reshape(1, 3072)
    res = np.argmax((model.predict(img)))
    return res

def nn_sparse(X_all, Y, isTrain, activation, optimizer, loss, epochs, batch_size, lr, isGray, save_dir, base_path, array_layers):
    if isGray:
        X_all = np.mean(X_all.reshape(-1, 32 * 32, 3), axis=2)
        image_size = 32 * 32
    else:
        image_size = 32 * 32 * 3
    nb_output = np.max(Y) + 1
    directory = base_path + save_dir
    if not os.path.exists(directory):
        os.mkdir(directory)
    path = directory + "/model.h5"
    if isTrain:
        model = nn_model(image_size, nb_output, activation, optimizer, loss, lr, array_layers)

        model = model_fit(model, X_all, Y,
                          epochs, batch_size,
                          path, save_dir, base_path)

