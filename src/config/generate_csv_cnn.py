# import pandas as pd 
import pandas as pd 
import random 
import numpy as np

#activation,optimizer,loss,epochs,batch-size,learning-rate,layers,kernel-shape,pooling,isGray,isSparse,isNorm,Dropout,L1,L2



activations=["linear", "sigmoid", "hard_sigmoid", "relu", "linear", "tanh", "elu", "selu", "softmax", "softplus", "softsign"]
optimizers=["adadelta", "adagrad", "adam", "adamax", "ftrl", "nadam", "rmsprop", "sgd"]
losses=["categorical_crossentropy"]#["categorical_hinge", "categorical_crossentropy", "kullback_leibler_divergence"]
batchs=[1000, 5000] #[500, 1000, 3000, 5000, 10000, 60000]
epochs=[500]#[50, 100, 200, 300]
lrs=[0.0001]
bool_l=[False]
l1l2 = [0]
dropout = [0]
kernel_shape = [2]
nb_neurons = [24]
nb_layers = [4]
pooling = ["max_pool", "avg_pool"]

activation_param = []
optimizer_param = []
loss_param = []
batch_size_param = []
epochs_param = []
lr = []
isGray_param = []
isNorm_param = []
kernel_shape_param = []
pooling_param = []
dropout_param = []
l1_param = []
l2_param = []
layers_param = []

for i in range(10000):
    activation_param.append(random.choice(activations))
    optimizer_param.append(random.choice(optimizers))
    loss_param.append(random.choice(losses))
    batch_size_param.append(random.choice(batchs))
    epochs_param.append(random.choice(epochs))
    lr.append(random.choice(lrs))
    isGray_param.append(random.choice(bool_l))
    isNorm_param.append(random.choice(bool_l))

    kernel_shape_param.append(random.choice(kernel_shape))
    pooling_param.append(random.choice(pooling))
    dropout_param.append(random.choice(dropout))
    l1_param.append(random.choice(l1l2))
    l2_param.append(random.choice(l1l2))

    layers = []
    same_activation = random.choice(activations)
    for j in range(random.choice(nb_layers)):
        layers.append(random.choice(nb_neurons))

    layers.sort(reverse=True)
    layers = '-'.join(map(str, layers))
    layers_param.append(layers)

# Calling DataFrame constructor after zipping 
# both lists, with columns specified 
l = list(zip(activation_param, optimizer_param, loss_param, epochs_param, batch_size_param,
                lr, layers_param, kernel_shape_param, pooling_param, isGray_param,
                isNorm_param, dropout_param, l1_param, l2_param))
list_set = set(l) 
unique_list = (list(list_set))
df = pd.DataFrame(unique_list, 
               columns =["activation", "optimizer", "loss", "epochs", "batch-size", "learning-rate", "layers", "kernel-shape", "pooling", "isGray", "isNorm", "Dropout", "L1", "L2"]) 
df.to_csv("../../config/cnn.csv", index=False)