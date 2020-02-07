import os
import shutil

# def merge export
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
from tensorflow.keras import Model

from utils.data import load_validation_data
from utils.tools import generate_name

# def merge_exports():

def export_tensorboard_regularizers(source, dest, model_path, X, Y):
    X_val, Y_val = load_validation_data()
    with open(source) as input:
        configs = input.readlines()
        with open(dest, "w") as output:
            output.write(configs[0][0:-1] + ",last_loss, last_val_loss, last_accuracy, last_val_accuracy\n")
            for conf in configs[1::]:
                filename = generate_name(conf.split(","))

                if filename[-1] == "\n":
                    filename = filename[0:-1] + "_sparse"
                else:
                    filename = filename + "_sparse"

                # #train_path = model_path + filename + "\\train\\"
                val_path = model_path + filename + "\\validation\\"
                #
                # #ea = event_accumulator.EventAccumulator(path=train_path)
                ea_val = event_accumulator.EventAccumulator(path=val_path)
                # #ea.Reload()
                ea_val.Reload()
                # #loss = ea.Scalars('epoch_loss')[-1][2]
                loss_val = ea_val.Scalars('epoch_loss')[-1][2]
                # #accuracy = ea.Scalars('epoch_sparse_categorical_accuracy')[-1][2]
                accuracy_val = ea_val.Scalars('epoch_sparse_categorical_accuracy')[-1][2]


                fullPath = model_path + filename + "/model.h5"
                model = tf.keras.models.load_model( fullPath)
                result_train = model.evaluate(X, Y, batch_size=5000, verbose=0)
                # = model.evaluate(X_val, Y_val, batch_size=5000, verbose=0)

                if conf[-1] == "\n":
                    conf = conf[0:-1]
                output.write(conf + "," + str(round(result_train[0], 5)) + "," + str(round(loss_val, 5)) + "," + str(round(result_train[1], 5)) + "," + str(round(accuracy_val, 5)) + "\n")

def export_tensorboard_to_csv(source, dest, model_path):
    with open(source) as input:
        configs = input.readlines()
        with open(dest, "w") as output:
            output.write(configs[0][0:-1] + ",last_loss, last_val_loss, last_accuracy, last_val_accuracy\n")
            for conf in configs[1::]:
                filename = generate_name(conf.split(","))

                if filename[-1] == "\n":
                    filename = filename[0:-1] + "_sparse"
                else:
                    filename = filename + "_sparse"

                train_path = model_path + filename + "\\train\\"
                val_path = model_path + filename + "\\validation\\"

                ea = event_accumulator.EventAccumulator(path=train_path)
                ea_val = event_accumulator.EventAccumulator(path=val_path)
                ea.Reload()
                ea_val.Reload()
                loss = ea.Scalars('epoch_loss')[-1][2]
                loss_val = ea_val.Scalars('epoch_loss')[-1][2]
                accuracy = ea.Scalars('epoch_sparse_categorical_accuracy')[-1][2]
                accuracy_val = ea_val.Scalars('epoch_sparse_categorical_accuracy')[-1][2]

                if conf[-1] == "\n":
                    conf = conf[0:-1]
                output.write(conf + "," + str(round(loss, 5)) + "," + str(round(loss_val, 5)) + "," + str(round(accuracy, 5)) + "," + str(round(accuracy_val, 5)) + "\n")
