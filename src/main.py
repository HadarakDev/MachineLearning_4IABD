import numpy as np
import matplotlib.pyplot as plt
from tools import unpickle, get_label_names, display_batch_stat
from linear10Models import linear_model, load_linear_model, predict_liear
import tensorflow as tf
from multiprocessing import Pool


datasetPath = "../dataset/data_batch_"
size = 32
image_size = 32 * 32 * 3
isTrain = False

label_names = get_label_names()
#for i in range(1, 5):
#    display_batch_stat(i, label_names, datasetPath)

features, labels = unpickle("../dataset/data_batch_1", size, True)

X_all = features.flatten().reshape(10000, size * size * 3)
Y = np.asarray(labels)

Y_all = []
model_all = []

for i in range(10):
    Y_all.append(np.array([1 if y == i else 0 for y in Y]))

if isTrain:
    for i in range(10):
        model_all.append(linear_model(X_all, Y_all[i],
            "sigmoid",
            "adam",
            "binary_crossentropy",
            10000,
            200,
            "../models/linear10models/model" + str(i) + ".h5",
            image_size)
        )
else:
    for i in range(10):
        model_all.append(load_linear_model("../models/linear10models/model" + str(i) + ".h5"))

#for i in range(10):
#    model_all[i].summary()


# params = []
# for i in range(10):
#     params.append((X_all[i]))

# def result_thread(X):
#     return predict_liear(model_all, X)

# p = Pool()
# result = p.starmap(result_thread, params)
# p.close()
# p.join()   

result= [] 
range_img = 500
for i in range(range_img):
    result.append(predict_liear(model_all, X_all[i]))

res_stats = []

for i in range(range_img):
    if result[i] == Y[i]:
        res_stats.append(1)
    else:
        res_stats.append(0)

print(str((sum(res_stats) / range_img) * 100) + "%")


#print("Image predicted is a: " + label_names[res])
#print("Image is supposed to be a: " + label_names[Y[img_index]])