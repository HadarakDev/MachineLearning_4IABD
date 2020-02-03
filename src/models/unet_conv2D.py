import os

from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.metrics import *
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow_core.python.keras.regularizers import L1L2

from src.utils.tools import unpickle, get_optimizer


def create_model(activation_param, optimizer_param, lr_param, loss_param, array_layers, kernel_shape_param, depth):
    optimizer_param = get_optimizer(optimizer_param, lr_param)
    stack = []
    input_layer = Input((32, 32, 3))
    # Edit padding depending of size of image. For cifar 10, do not zeropad
    padding_layer = ZeroPadding2D((2, 2))(input_layer)
    last_output = Conv2D(filters=array_layers[0],
                         kernel_size=(kernel_shape_param, kernel_shape_param),
                         activation=activation_param,
                         name=f"Conv2D",
                         padding='SAME')(padding_layer)
    stack.append(last_output)

    last_output = MaxPooling2D((2, 2), padding='SAME', name=f"MaxPooling2D")(last_output)

    for i in range((depth * 2)):
        last_output = Conv2D(filters=array_layers[i],
                             kernel_size=(kernel_shape_param, kernel_shape_param),
                             activation=activation_param,
                             name=f"Conv2D_{i}",
                             padding='SAME')(last_output)
        # last_output = BatchNormalization(name=f"BatchNormalization_{i}")(last_output)
        # last_output = Activation(activation=relu, name=f"Activation_{i}")(last_output)
        if i < depth:
            stack.append(last_output)
            last_output = MaxPooling2D((2, 2), padding='SAME', name=f"MaxPooling2D_{i}")(last_output)
        else:
            last_output = UpSampling2D((2, 2), name=f"UpSampling2D_{i}")(last_output)
            last_output = Add(name=f"Add_{i}")([last_output, stack.pop()])

    last_output = Conv2D(filters=array_layers[len(array_layers) - 1],
                         kernel_size=(kernel_shape_param, kernel_shape_param),
                         activation=activation_param,
                         name=f"Conv2D_last",
                         padding='SAME')(last_output)
    last_output = UpSampling2D((2, 2), name=f"UpSampling2D_last")(last_output)
    last_output = Add(name=f"Add_last")([last_output, stack.pop()])

    print(len(stack))
    last_output = Flatten(name="flatten")(last_output)
    # kernel_regularizer_param = get_kernel_regularizer()... kernel_regularizer=L1L2(l2=0.001 / depth)
    output_tensor = Dense(10, activation=softmax, name=f"Dense_output")(last_output)
    model = Model(input_layer, output_tensor)

    model.compile(loss=loss_param,
                  optimizer=optimizer_param,
                  metrics=[sparse_categorical_accuracy])
    return model


def unet_conv2D(X_all, Y, isTrain, activation_param, optimizer_param, lr_param, loss_param, batch_size_param,
                epochs_param, save_path_info, array_layers, kernel_shape_param):
    depth = int(len(array_layers) / 2)
    if depth % 2 == 0:
        depth = depth - 1

    model = create_model(activation_param, optimizer_param, lr_param, loss_param, array_layers, kernel_shape_param,
                         depth)
    print(model.summary())
    #plot_model(model, "unet_conv2d.png")

    # directory = "../models/Unet_conv2D/" + save_path_info
    # if not os.path.exists(directory):
    #     os.mkdir(directory)
    # path = directory + "/" + save_path_info + ".h5"

    X = X_all.reshape(50000, 32, 32, 3)
    model.fit(X, Y, validation_split=0.2,
              epochs=epochs_param,
              batch_size=batch_size_param)
