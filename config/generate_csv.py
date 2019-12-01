# import pandas as pd 
import pandas as pd 
import random 
  
activations=["sigmoid", "hard_sigmoid", "relu", "linear", "tanh", "elu", "selu", "softmax", "softplus", "softsign"]
optimizers=["adadelta", "adagrad", "adam", "adamax", "ftrl", "nadam", "rmsprop", "sgd"]
losses=["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "squared_hinge", "hinge", "categorical_hinge", "logcosh", "binary_crossentropy"]
batchs=[500, 1000, 3000, 5000, 10000, 60000]
epochs= [10, 20, 30]

activation_param = []
optimizer_param = []
loss_param = []
batch_size_param = []
epochs_param = []
save_path_info = []

for i in range(50):
    activation_param.append(random.choice(activations))
    optimizer_param.append(random.choice(optimizers))
    loss_param.append(random.choice(losses))
    batch_size_param.append(random.choice(batchs))
    epochs_param.append(random.choice(epochs))

    save_path_info.append(str(activation_param[i]) + "_" +
                           str(optimizer_param[i]) + "_" +
                           str(loss_param[i]) + "_" +
                           str(batch_size_param[i]) + "_" +
                           str(epochs_param[i]))

  
# Calling DataFrame constructor after zipping 
# both lists, with columns specified 
df = pd.DataFrame(list(zip(activation_param, optimizer_param, loss_param, batch_size_param, epochs_param, save_path_info)), 
               columns =["activation_param", "optimizer_param", "loss_param", "batch_size_param", "epochs_param", "save_path_info"]) 
df.to_csv("./linear_X.csv", index=False)