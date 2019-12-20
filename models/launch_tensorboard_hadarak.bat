
call activate tfGPU
start chrome http://localhost:6006/
tensorboard --logdir="./"

call conda deactivate