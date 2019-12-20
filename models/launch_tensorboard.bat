call activate tfmx150
start chrome http://localhost:6006/
tensorboard --logdir="./"
call conda deactivate