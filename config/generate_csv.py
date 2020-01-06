# import pandas as pd 
import pandas as pd 
import random 
import numpy as np
  
activations=["linear"]#["sigmoid", "hard_sigmoid", "relu", "linear", "tanh", "elu", "selu", "softmax", "softplus", "softsign"]
optimizers=["adadelta", "adagrad", "adam", "adamax", "ftrl", "nadam", "rmsprop", "sgd"]
losses=["categorical_crossentropy"]#["categorical_hinge", "categorical_crossentropy", "kullback_leibler_divergence"]
batchs=[5000]#[500, 1000, 3000, 5000, 10000, 60000]
epochs=[200]#[50, 100, 200, 300]
lrs=[0.0001]

activation_param = []
optimizer_param = []
loss_param = []
batch_size_param = []
epochs_param = []
save_path_info = []
lr = []

for i in range(100):
    activation_param.append(random.choice(activations))
    optimizer_param.append(random.choice(optimizers))
    loss_param.append(random.choice(losses))
    batch_size_param.append(random.choice(batchs))
    epochs_param.append(random.choice(epochs))
    lr.append(random.choice(lrs))

    save_path_info.append(str(activation_param[i]) + "_" +
                           str(optimizer_param[i]) + "_" +
                           str(lr[i]) + "_" +
                           str(loss_param[i]) + "_" +
                           str(batch_size_param[i]) + "_" +
                           str(epochs_param[i]))


# Calling DataFrame constructor after zipping 
# both lists, with columns specified 
l = list(zip(activation_param, optimizer_param, lr, loss_param, batch_size_param, epochs_param, save_path_info))
list_set = set(l) 
unique_list = (list(list_set))
df = pd.DataFrame(unique_list, 
               columns =["activation_param", "optimizer_param", "learning_rate", "loss_param", "batch_size_param", "epochs_param", "save_path_info"]) 
df.to_csv("./linear.csv", index=False)