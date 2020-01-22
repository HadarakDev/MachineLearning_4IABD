import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

def get_callbacks(log_dir):
    #Tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #Avoid overfit from accuracy 
    earlystop_callback = EarlyStopping(monitor='sparse_categorical_accuracy', min_delta=0.0001, patience=5)
    earlystop_val_callback = EarlyStopping(monitor='val_sparse_categorical_accuracy', min_delta=0.0001, patience=5)
    return [tensorboard_callback, earlystop_callback, earlystop_val_callback]