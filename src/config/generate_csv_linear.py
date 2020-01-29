# import pandas as pd 
import pandas as pd 
import random 
import numpy as np
  
activations=["linear", "sigmoid", "hard_sigmoid", "relu", "linear", "tanh", "elu", "selu", "softmax", "softplus", "softsign"]
optimizers=["adadelta", "adagrad", "adam", "adamax", "ftrl", "nadam", "rmsprop", "sgd"]
losses=["categorical_crossentropy"]#["categorical_hinge", "categorical_crossentropy", "kullback_leibler_divergence"]
batchs=[1000, 5000] #[500, 1000, 3000, 5000, 10000, 60000]
epochs=[500]#[50, 100, 200, 300]
lrs=[0.0001]
bool_l=[False, True]

activation_param = []
optimizer_param = []
loss_param = []
batch_size_param = []
epochs_param = []
lr = []
isGray_param =[]
isSparse_param = []
isNorm_param = []

for i in range(10000):
    activation_param.append(random.choice(activations))
    optimizer_param.append(random.choice(optimizers))
    loss_param.append(random.choice(losses))
    batch_size_param.append(random.choice(batchs))
    epochs_param.append(random.choice(epochs))
    lr.append(random.choice(lrs))
    isGray_param.append(random.choice(bool_l))
    isNorm_param.append(random.choice(bool_l))


# Calling DataFrame constructor after zipping 
# both lists, with columns specified 
l = list(zip(activation_param, optimizer_param, loss_param, epochs_param, batch_size_param, lr, isGray_param, isNorm_param))
list_set = set(l) 
unique_list = (list(list_set))
df = pd.DataFrame(unique_list, 
               columns =["activation", "optimizer", "loss", "epochs", "batch-size", "learning-rate", "isGray", "isNorm"]) 
df.to_csv("../../config/linear.csv", index=False)