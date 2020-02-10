from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.initializers import *
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.datasets import *
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow_core.python.keras.regularizers import L1L2
import numpy as np


from tools import unpickle, get_label_names, display_batch_stat, create_dirs
from config import isGray


def shift_out(last_output, penultimate_output, i):
    tmp_out = last_output
    last_output = Add(name=f"Add_{i}")([last_output, penultimate_output])
    penultimate_output = tmp_out
    return last_output, penultimate_output

def create_resnet_model_dense(X_all, Y, isTrain,  activation, optimizer, loss, epochs, batch_size, lr, isGray, save_dir, base_path, array_layers, jumps, dropout, l1, l2):
    input_layer = Input(shape)
    flatten_layer_output = Flatten(name="flatten")(input_layer)

    penultimate_output = None
    last_output = flatten_layer_output

    for i in range(1, len(array_layers)):
        if penultimate_output is not None:

            last_output, penultimate_output = shift_out(last_output, penultimate_output, i)
            for j in range(jumps):
                last_output = Dense(array_layers[i], activation=linear, name=f"Dense_{i}_{j}", kernel_regularizer=L1L2(l2=0.001 / depth))(last_output)
                last_output = BatchNormalization(name=f"BatchNormalization_{i}_{j}")(last_output)
                last_output = Activation(activation=relu, name=f"Activation_{i}_{j}")(last_output)
        else:
            penultimate_output = last_output
            for j in range(jumps):
                last_output = Dense(array_layers[i], activation=linear, name=f"Dense_{i}_{j}", kernel_regularizer=L1L2(l2=0.001 / depth))(last_output)
                last_output = BatchNormalization(name=f"BatchNormalization_{i}_{j}")(last_output)
                last_output = Activation(activation=relu, name=f"Activation_{i}_{j}")(last_output)

    
    last_output = Add(name=f"Add_output")([last_output, penultimate_output])

    output_tensor = Dense(10, activation=softmax, name=f"Dense_output", kernel_regularizer=L1L2(l2=0.001 / depth))(last_output)
    model = Model(input_layer, output_tensor)

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(lr=0.001),
                  metrics=[sparse_categorical_accuracy])
    return model


def resnet_model_dense(X_all, Y, isTrain,  activation, optimizer, loss, epochs, batch_size, lr, isGray, save_dir, base_path, array_layers, jumps, dropout, l1, l2):
    model = create_resnet_model_dense(X_all, Y, True,  activation, optimizer, loss, epochs, batch_size, lr, isGray, save_dir, base_path, array_layers, jumps, dropout, l1, l2)


if __name__ == "__main__":
    datasetPath = "../dataset/data_batch_"
    size = 32
    isTrain = True

    label_names = get_label_names()
    #for i in range(1, 5):
    #    display_batch_stat(i, label_names, datasetPath)

    X_all = []
    Y = []

    for i in range(1, 6):
        features, labels = unpickle("../dataset/data_batch_{}".format(i), size, True)
        X_all.append(features.flatten().reshape(10000, size * size * 3))
        Y.append(np.asarray(labels))

    X_all = np.concatenate(X_all)
    # Gray
    if isGray:
        X_all = np.mean(X_all.reshape(-1, size * size, 3), axis=2)
        X_all = X_all.reshape(50000, size, size)
    else:
        X_all = X_all.reshape(50000, size, size, 3)
    # norm
    X_all = X_all / 255.0
    
    print(np.shape(X_all))
    Y = np.concatenate(Y)

    model = resnet_model_dense(5, 2, (size, size))
    print(model.summary())
    plot_model(model, "residual_dense.png")
    model.fit(X_all, Y, validation_split=0.2,
              epochs=100,
              batch_size=5000)
