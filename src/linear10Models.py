import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


def linear_model(X_param, Y_param, activation_param, optimizer_param, loss_param, batch_size_param, epochs_param, save_path, size):
    model = tf.keras.Sequential()
    model.add(Dense(1, activation=activation_param, input_dim=size))
    model.compile(optimizer=optimizer_param, loss=loss_param, metrics=["accuracy"])
    model.fit(X_param, Y_param, batch_size=batch_size_param, verbose=1, epochs=epochs_param)
    model.save(save_path) 
    return model

def predict_liear(model_all, X):
    res = []
    for model in model_all:
        img = X.reshape(1, 3072)
        res.append(model.predict(img)[0][0])
    print(res)
    return res.index(max(res))

def load_linear_model(model_path):
    return tf.keras.models.load_model(model_path)