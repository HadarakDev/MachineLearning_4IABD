import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

def get_callbacks(log_dir):
    #Tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #Avoid overfit from accuracy 
    earlystop_callback = EarlyStopping(monitor='sparse_categorical_accuracy', min_delta=0.0001, patience=5)
    earlystop_val_callback = EarlyStopping(monitor='val_sparse_categorical_accuracy', min_delta=0.0001, patience=5)
    return [tensorboard_callback, earlystop_callback, earlystop_val_callback]

def model_fit(model, X_param, Y_param, batch_size_param, epochs_param, save_path, save_path_info):
    log_dir = "..\\models\\cnn_sparse\\" + save_path_info
    call_backs = get_callbacks(log_dir)
    model.fit(X_param, Y_param, batch_size=batch_size_param, verbose=1, epochs=epochs_param, callbacks=call_backs, validation_split=0.2)
    model.save(save_path)
    return model