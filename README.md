Implementation of improvement of Fixmatch

To test the provided models in the model folder for CIFAR-10 250 and 4000 labels and for CIFAR-100 2500 and 10000 labels, please provide the argument for train.py of appropriate model file name and the second argument of flag to set in test mode.
Along with the third argument of which dataset to be choosen("cifar10"or"cifar100")
(eg train.py --bestmodel "./result/model_best.pth.tar" --train-mode 0 --dataset "cifar100")

Taken some of the code like misc,ema and dataloader from https://github.com/kekmodel/FixMatch-pytorch


Models saved:

model_best_sim_10000_cf100.pth.tar
model_best_sim_4000_cf10.pth.tar
model_best_sim_2500_cf100.pth.tar
model_best_sim_250_cf10.pth.tar