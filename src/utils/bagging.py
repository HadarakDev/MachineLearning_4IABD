import numpy as np

from sklearn.metrics import accuracy_score


# model_list : list of 'best' models
# X : train/test/validation X
# Y : train/test/validation Y
# Returns the accuracy_score
def bagging_prediction(model_list, X, Y):
	res = [model.predict(X) for model in model_list]
	res = array(res)
	res_sum = np.sum(res, axis=0)
	res_best = np.argmax(res_sum, axis=1)
  return accuracy_score(Y, res_best)
