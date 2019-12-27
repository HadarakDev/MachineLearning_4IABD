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

def create_model(depth):
    stack = []
    input_layer = Input((28, 28, 3))
    last_output = Conv2D(100, (3, 3), activation='relu', name=f"Conv2D", padding='SAME')(input_layer)
    stack.append(last_output)

    for i in range((depth * 2) + 1):
        last_output = Conv2D(100, (3, 3), activation='relu', name=f"Conv2D_{i}", padding='SAME')(last_output)
        #last_output = BatchNormalization(name=f"BatchNormalization_{i}")(last_output)
        #last_output = Activation(activation=relu, name=f"Activation_{i}")(last_output)
        if i < depth:
            stack.append(last_output)
        else:
            last_output = Add(name=f"Add_{i}")([last_output, stack.pop()])

    last_output = Flatten(name="flatten")(last_output)
    output_tensor = Dense(10, activation=softmax, name=f"Dense_output", kernel_regularizer=L1L2(l2=0.001 / depth))(last_output)
    model = Model(input_layer, output_tensor)

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(lr=0.001),
                  metrics=[sparse_categorical_accuracy])
    return model


if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val) = fashion_mnist.load_data()
    x_train = x_train / 255.0
    x_val = x_val / 255.0
    model = create_model(1)
    print(model.summary())
    plot_model(model, "unet_conv2d.png")

    # only for fashion_mnist. Gray Scale to "RBG"
    x_train = stacked_img = np.stack((x_train,) * 3, axis=-1)


    model.fit(x_train, y_train, validation_split=0.2,
              epochs=100,
              batch_size=800)
