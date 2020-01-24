
call activate tfGPU
start chrome http://localhost:6006/
tensorboard --logdir="./sparse_vs_oneHot_Linear/"

call conda deactivate