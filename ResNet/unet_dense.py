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

def create_model(depth):
    input_layer = Input((28, 28))
    flatten_layer_output = Flatten(name="flatten")(input_layer)

    stack = []
    last_output = flatten_layer_output

    for i in range(depth * 2):
        last_output = Dense(784, activation=linear, name=f"Dense_{i}", kernel_regularizer=L1L2(l2=0.001 / depth))(last_output)
        #last_output = BatchNormalization(name=f"BatchNormalization_{i}")(last_output)
        #last_output = Activation(activation=relu, name=f"Activation_{i}")(last_output)
        if i < depth:
            stack.append(last_output)
        else:
            last_output = Add(name=f"Add_{i}")([last_output, stack.pop()])

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
    model = create_model(3)
    print(model.summary())
    plot_model(model, "unet.png")
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=100,
              batch_size=8192)
