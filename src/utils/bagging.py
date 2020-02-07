import numpy as np

from sklearn.metrics import accuracy_score
from tensorflow_core.python.keras.saving import load_model

from utils.tools import generate_name


# model_list : list of 'best' models
# X : train/test/validation X
# Y : train/test/validation Y
# Returns the accuracy_score
def bagging_prediction(model_list, paths, X, Y):
    res = []
    length = len(model_list)
    X_cnn = X.reshape(10000, 32, 32, 3)
    X_cnn_norm = X_cnn / 255.0
    X_norm = X / 255.0
    for i, model in enumerate(model_list):
        print("Model " + str(i + 1) + " over " + str(length))
        if "Cnn" in paths[i] and "False_True" in paths[i]:
            res.append(model.predict(X_cnn_norm))
        elif "Cnn" in paths[i]:
            res.append(model.predict(X_cnn))
        elif "False_True" in paths[i]:
            res.append(model.predict(X_norm))
        else:
            res.append(model.predict(X))


    res = np.array(res)
    res_sum = np.sum(res, axis=0)
    print("summed results")
    res_best = np.argmax(res_sum, axis=1)
    print("Best results, computing accuracy score")
    return accuracy_score(Y, res_best)


def get_models_from_csv(source, model_path, X, Y):
    models = []
    paths = []
    with open(source) as input:
        configs = input.readlines()
        for conf in configs[1::]:
            filename = generate_name(conf.split(","))

            if filename[-1] == "\n":
                filename = filename[0:-1] + "_sparse"
            else:
                filename = filename + "_sparse"

            path = model_path + filename + "/model.h5"
            paths.append(path)
            print(path)
            model = load_model(path)
            models.append(model)

    X_all = X.astype(np.float32)
    print(bagging_prediction(models, paths, X_all, Y))