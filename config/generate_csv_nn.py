# import pandas as pd 
import pandas as pd 
import random 
  
activations=["sigmoid", "hard_sigmoid", "relu", "linear", "tanh", "elu", "selu", "softmax", "softplus", "softsign"]
optimizers=["adadelta", "adagrad", "adam", "adamax", "ftrl", "nadam", "rmsprop", "sgd"]
losses=["categorical_crossentropy"]
batchs=[5000]
epochs=[200]
lrs=[0.0001]

activation_param = []
optimizer_param = []
loss_param = []
batch_size_param = []
epochs_param = []
save_path_info = []
array_layers = []
lr = []

max_rand = 1000

for i in range(100000):
    optimizer_param.append(random.choice(optimizers))
    loss_param.append(random.choice(losses))
    batch_size_param.append(random.choice(batchs))
    epochs_param.append(random.choice(epochs))
    lr.append(random.choice(lrs))

    layers = []
    activ = []
    same_activation = random.choice(activations)
    for j in range(4):
        activ.append(same_activation)
        layers.append(random.randrange(64, 65, 1))
    
    
    layers.sort(reverse = True)
    activ = ';'.join(map(str, activ)) 
    layers = ';'.join(map(str, layers)) 

    activation_param.append(activ)
    array_layers.append(layers)
    save_path_info.append(str(activation_param[i]) + "_" +
                           str(optimizer_param[i]) + "_" +
                           str(lr[i]) + "_" +
                           str(loss_param[i]) + "_" +
                           str(batch_size_param[i]) + "_" +
                           str(epochs_param[i]))

  
# Calling DataFrame constructor after zipping 
# both lists, with columns specified 
l = list(zip(activation_param, optimizer_param, lr, loss_param, batch_size_param, epochs_param, save_path_info, array_layers))
list_set = set(l) 
unique_list = (list(list_set))
df = pd.DataFrame(list_set, 
               columns =["activation_param", "optimizer_param", "learning_rate", "loss_param", "batch_size_param", "epochs_param", "save_path_info", "array_layers"]) 
df.to_csv("./nn.csv", index=False)