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


def show_first_samples(x_train, y_train):
    plt.imshow(x_train[0])
    print(y_train[0])
    plt.show()
    plt.imshow(x_train[1])
    print(y_train[1])
    plt.show()
    plt.imshow(x_train[2])
    print(y_train[2])
    plt.show()
    plt.imshow(x_train[3])
    print(y_train[3])
    plt.show()


def create_model(depth: int = 34, use_skip_connections: bool = True):
    layers_config = []
    for i in range(3):
        layers_config.append(64)
    for i in range(4):
        layers_config.append(128)
    for i in range(6):
        layers_config.append(256)
    for i in range(3):
        layers_config.append(512)


    input_layer = Input((28, 28, 1))

    print(input_layer)
    conv_1 = Conv2D(64, (7, 7), padding="SAME", activation=relu)(input_layer)
    print(conv_1)
    #maxPool_1 = MaxPooling2D((2, 2), padding="SAME")(conv_1)
    #add_out_conv_1 = Add(name="conv_1")([conv_1, maxPool_1])


    pen_out = None
    third_out = None
    last_out = conv_1

    for i, neuron in enumerate(layers_config):
        print(neuron)
        if pen_out is not None and use_skip_connections:

            if layers_config[i - 1] != layers_config[i]:
                add_out = Add(name=f"Add_{i}a")([third_out, last_out])
                pen_out = add_out
                third_out = pen_out
                penultimate_out = last_out

                last_out = Dense(1, name=f"dense_{i}", activation=relu)

                last_out = Conv2D(neuron, (3, 3), activation=relu, padding="SAME", strides=(2, 2), name=f"Conv2D_strides_{i}")(add_out)


                last_out = Conv2D(neuron, (3, 3), activation=relu, padding="SAME", name=f"Conv2D_{i}")(add_out)
                add_out = Add(name=f"Add_{i}b")([pen_out, last_out])
            else:
                add_out = Add(name=f"Add_{i}a")([third_out, last_out])
                pen_out = add_out
                last_out = Conv2D(neuron, (3, 3), activation=relu, padding="SAME",  name=f"Conv2D_{i}")(add_out)

                last_out = Conv2D(neuron, (3, 3), activation=relu, padding="SAME", name=f"Conv2D_{i}")(add_out)
                add_out = Add(name=f"Add_{i}b")([pen_out, last_out])
        else:
            third_out = last_out
            last_out = Conv2D(neuron, (3, 3), activation=relu, padding="SAME", name=f"Conv2D_{i}")(last_out)

            pen_out = last_out
            last_out = Conv2D(neuron, (3, 3), activation=relu, padding="SAME", name=f"Conv2D_{i}")(last_out)
            add_out = Add(name=f"Add_{i}")([pen_out, last_out])

    if use_skip_connections:
        last_out = Add(name=f"add_out")([pen_out, last_out])

    output_tensor = Dense(10, activation=softmax, name=f"Dense_output")(last_out)
    model = Model(input_layer, output_tensor)

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=[sparse_categorical_accuracy])
    return model


if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val) = fashion_mnist.load_data()
    x_train = x_train / 255.0
    x_val = x_val / 255.0
    show_first_samples(x_train, y_train)
    model = create_model(use_skip_connections=True)
    print(model.summary())
    plot_model(model, "residual_dense.png")
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=100,
              batch_size=8192)
