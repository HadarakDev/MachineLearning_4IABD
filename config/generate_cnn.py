import pandas as pd
import random

activations=["sigmoid", "hard_sigmoid", "relu", "linear", "tanh", "elu", "selu", "softmax", "softplus", "softsign"]
optimizers=["adadelta", "adagrad", "adam", "adamax", "ftrl", "nadam", "rmsprop", "sgd"]
pooling=["avg_pool", "max_pool"]
losses=["categorical_crossentropy"]
kernel_shape=[2, 3]
batchs=[500]
epochs=[600]
lrs=[0.001]

activation_param = []
optimizer_param = []
pooling_param = []
kernel_shape_param = []
loss_param = []
batch_size_param = []
epochs_param = []
save_path_info = []
array_layers = []
lr = []

for i in range(10000):
    optimizer_param.append(random.choice(optimizers))
    pooling_param.append(random.choice(pooling))
    kernel_shape_param.append(random.choice(kernel_shape))
    loss_param.append(random.choice(losses))
    batch_size_param.append(random.choice(batchs))
    epochs_param.append(random.choice(epochs))
    lr.append(random.choice(lrs))

    layers = []
    activ = []
    same_activation = random.choice(activations)
    for j in range(4):
        activ.append(same_activation)
        layers.append(random.randrange(128, 129, 1))

    layers.sort(reverse=True)
    activ = ';'.join(map(str, activ))
    layers = ';'.join(map(str, layers))

    activation_param.append(activ)
    array_layers.append(layers)
    save_path_info.append(str(activation_param[i]) + "_" +
                          str(optimizer_param[i]) + "_" +
                          str(pooling_param[i]) + "_" +
                          str(kernel_shape_param[i]) + "_" +
                          str(lr[i]) + "_" +
                          str(loss_param[i]) + "_" +
                          str(batch_size_param[i]) + "_" +
                          str(epochs_param[i]) + "_" +
                          str(layers))

    l = list(zip(activation_param, optimizer_param, pooling_param, lr, loss_param, batch_size_param, epochs_param, save_path_info,
                 array_layers, kernel_shape_param))
    list_set = set(l)
    unique_list = (list(list_set))
    df = pd.DataFrame(list_set,
                      columns=["activation_param", "optimizer_param", "pooling_param", "learning_rate", "loss_param", "batch_size_param",
                               "epochs_param", "save_path_info", "array_layers", "kernel_shape_param"])
    df.to_csv("./cnn.csv", index=False)