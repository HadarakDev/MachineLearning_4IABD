# import pandas as pd 
import pandas as pd 
import random 
  
activations=["relu", "sigmoid"]
optimizers=["adam", "adamax"]
losses=["binary_crossentropy"]
batchs=[500, 1000, 3000, 5000, 10000, 60000]
epochs=[50, 100, 200, 300]

activation_param = []
optimizer_param = []
loss_param = []
batch_size_param = []
epochs_param = []
save_path_info = []
array_layers = []
for i in range(10):
    optimizer_param.append(random.choice(optimizers))
    loss_param.append(random.choice(losses))
    batch_size_param.append(random.choice(batchs))
    epochs_param.append(random.choice(epochs))

    activ = ""
    layers = ""
    for j in range(random.randint(3, 10)):
        activ = activ + random.choice(activations) + ";"
        layers = layers + str(random.randrange(10, 100, 10)) + ";"
    activ = activ + random.choice(activations)
    layers = layers + str(random.randrange(10, 100, 10))

    activation_param.append(activ)
    array_layers.append(layers)
    save_path_info.append(str(activation_param[i]) + "_" +
                           str(optimizer_param[i]) + "_" +
                           str(loss_param[i]) + "_" +
                           str(batch_size_param[i]) + "_" +
                           str(epochs_param[i]))

  
# Calling DataFrame constructor after zipping 
# both lists, with columns specified 
df = pd.DataFrame(list(zip(activation_param, optimizer_param, loss_param, batch_size_param, epochs_param, save_path_info, array_layers)), 
               columns =["activation_param", "optimizer_param", "loss_param", "batch_size_param", "epochs_param", "save_path_info", "array_layers"]) 
df.to_csv("./nn.csv", index=False)