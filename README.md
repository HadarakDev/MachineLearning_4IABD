# MachineLearning_4IABD
Projet Machine Learning 4 IABD

 - Etude du dataset CIFAR-10
 - Etude des datasets Kaggle


# Linear

Best  5  models:

| Name                                                                   | Val_accuracy      | Val_loss         |
|------------------------------------------------------------------------|-------------------|------------------|
| linear_adam_0.0001_categorical_crossentropy_5000_200_norm              | 0.187600002       | 2.155102015      |
| sigmoid_nadam_0.0001_sparse_categorical_crossentropy_5000_200_norm     | 0.180000007       | 2.22626543       |
| linear_adamax_0.0001_categorical_crossentropy_5000_200_norm            | 0.177599996       | 2.117193222      |
| linear_nadam_0.0001_sparse_categorical_crossentropy_5000_200_gray_norm | 0.177599996328354 | 2.21883940696716 |
| selu_rmsprop_0.0001_categorical_crossentropy_5000_200_gray_norm        | 0.177100002765656 | 2.20988273620605 |

# NN without Regularizers

Best 5  models:

# NN with Regularizers

# TODO

Pour run un NN =>
1) Trouver les top configs dans le results.csv
2) Le remettre dans le csv => config\nn.csv
3) Changer les noms (rajouter dropout (0.2 ou 0.4) dans le nom et/ou l1l2)
4) Editer le fichier nn_sparse.py pour rajouter le dropout ou l1l2 
5) run



DROPOUT 0.2 :

DROPOUT 0.4 :

L1l2  :

L1l2 DROPOUT 0.2 :

L1l2 DROPOUT 0.4 :

REDUCE NN NB NEURONS PER LAYERS : Nico
