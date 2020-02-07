import pickle
import tensorflow as tf
import numpy as np

from utils.tools import unpickle

def load_validation_data():
    X_all = []
    Y = []

    features, labels = unpickle("../dataset/test_batch", 32, True)
    X_all.append(features.flatten().reshape(10000, 32 * 32 * 3))
    Y.append(np.asarray(labels))

    return X_all, Y

def load_dataset():
    tf.random.set_seed(1)
    size = 32
    X_all = []
    Y = []

    for i in range(1, 6):
        features, labels = unpickle("../dataset/data_batch_{}".format(i), size, True)
        X_all.append(features.flatten().reshape(10000, size * size * 3))
        Y.append(np.asarray(labels))

    X_all = np.concatenate(X_all)
    Y = np.concatenate(Y)
    return X_all, Y

def display_batch_stat(batch_nb, label_names, datasetPath, size, isRGB):
    features, labels = unpickle(datasetPath + str(batch_nb), size, isRGB)
    print("Batch N° %s" % str(batch_nb), "\n")
    print("Number of Samples in batch %s" % str(len(features)), "\n")
    counts = [[x, labels.count(x)] for x in set(labels)]
    for c in counts:
        print( "%s = %d <=> %.2f %s" % (label_names[c[0]], c[1], (100 * c[1]) / len(features), "%"))

def display_all_batch_stat():
    label_names = get_label_names()
    for i in range(1, 5):
       display_batch_stat(i, label_names, "../dataset/data_batch_")

def get_label_names():
    with open("../dataset/batches.meta", 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return (data['label_names'])



