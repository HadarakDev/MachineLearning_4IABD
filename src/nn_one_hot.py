import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Model
import numpy as np
import os
from tools import unpickle, get_label_names, display_batch_stat, load_linear_model, get_optimizer, y_one_hot
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD
from tools import isGray

def nn_model(size, nb_output, activation_param, optimizer_param, lr_param, loss_param, array_layers):
    optimizer_param = get_optimizer(optimizer_param, lr_param)
    model = tf.keras.Sequential()

    model.add(Dense(array_layers[0], activation=activation_param[0], input_dim=size))

    for i in range(1, len(array_layers)):
        #model.add(Dropout(0.1))
        model.add(Dense(array_layers[i], activation=activation_param[i]))

    model.add(Dense(nb_output, activation="softmax"))
    model.compile(optimizer=optimizer_param, loss=loss_param, metrics=['accuracy'])
    model.summary()
    return model

def nn_model_fit(model, X_param, Y_param, batch_size_param, epochs_param, save_path, save_path_info):
    log_dir = "..\\models\\nn_one_hot\\" + save_path_info
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(X_param, Y_param, batch_size=batch_size_param, verbose=1, epochs=epochs_param, callbacks=[tensorboard_callback], validation_split=0.2)
    model.save(save_path)
    return model

def predict_nn(model, X):
    img = X.reshape(1, 3072)
    res = np.argmax((model.predict(img)))
    return res

def nn_one_hot(X_all, Y, isTrain,  activation_param, optimizer_param, lr_param, loss_param, batch_size_param, epochs_param, save_path_info, array_layers):
    Y_one_hot = y_one_hot(Y, max(Y) + 1)
    nb_output = np.shape(Y_one_hot)[1]
    if isGray:
        image_size = 32 * 32
    else:
        image_size = 32 * 32 * 3

    directory = "../models/nn_one_hot/" + save_path_info
    if not os.path.exists(directory):
        os.mkdir(directory)
    path = directory + "/" + save_path_info + ".h5"
    if isTrain:
        model = nn_model(
                    image_size,
                    nb_output,
                    activation_param,
                    optimizer_param,
                    lr_param,
                    loss_param,
                    array_layers)
        model = nn_model_fit(model, X_all,
                    Y,
                    batch_size_param,
                    epochs_param,
                    path, save_path_info)
    else:
        model = load_linear_model(path)
    
    #for i in range(1000):
    #    res = predict_nn(model, X_all[i])
    #    print("predict => " + label_names[res] + " | expected => "  + label_names[Y[i]])