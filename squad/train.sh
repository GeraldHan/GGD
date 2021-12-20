CUDA_VISIBLE_DEVICES=0 python train.py --name baseline
CUDA_VISIBLE_DEVICES=0 python train_GGE.py --name ce_decompose
