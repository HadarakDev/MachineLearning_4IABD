# import pandas as pd 
import pandas as pd 
import random 
  
activations=["relu", "sigmoid", "tanh"]
optimizers=["adam", "sgd"]
losses=["sparse_categorical_crossentropy"]
batchs=[500, 1000, 3000, 5000, 10000, 60000]
epochs=[400, 500, 600, 700]
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

for i in range(5):
    optimizer_param.append(random.choice(optimizers))
    loss_param.append(random.choice(losses))
    batch_size_param.append(random.choice(batchs))
    epochs_param.append(random.choice(epochs))
    lr.append(random.choice(lrs))

    layers = []
    activ = []
    for j in range(random.randint(5, 15)):
        activ.append(random.choice(activations))
        layers.append(random.randrange(100, max_rand, 100))
    
    
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
df = pd.DataFrame(list(zip(activation_param, optimizer_param, lr, loss_param, batch_size_param, epochs_param, save_path_info, array_layers)), 
               columns =["activation_param", "optimizer_param", "learning_rate", "loss_param", "batch_size_param", "epochs_param", "save_path_info", "array_layers"]) 
df.to_csv("./nn.csv", index=False)