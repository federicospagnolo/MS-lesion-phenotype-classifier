#!/bin/bash
export CUDA_VISIBLE_DEVICES=$4
python train_rimnet.py fold_3 $1 $2 $3 sub_fold_2 $5
python train_rimnet.py fold_3 $1 $2 $3 sub_fold_3 $5
python train_rimnet.py fold_4 $1 $2 $3 sub_fold_1 $5
python train_rimnet.py fold_4 $1 $2 $3 sub_fold_2 $5
python train_rimnet.py fold_4 $1 $2 $3 sub_fold_3 $5
python train_rimnet.py fold_5 $1 $2 $3 sub_fold_1 $5
python train_rimnet.py fold_5 $1 $2 $3 sub_fold_2 $5
python train_rimnet.py fold_5 $1 $2 $3 sub_fold_3 $5

#for rimnet
#sh train_2.sh T1 QSM 4 rimnet
#export CUDA_VISIBLE_DEVICES=$3
#python train_rimnet.py fold_3 $1 $2 sub_fold_2 $4
#python train_rimnet.py fold_3 $1 $2 sub_fold_3 $4
#python train_rimnet.py fold_4 $1 $2 sub_fold_1 $4
#python train_rimnet.py fold_4 $1 $2 sub_fold_2 $4
#python train_rimnet.py fold_4 $1 $2 sub_fold_3 $4
#python train_rimnet.py fold_5 $1 $2 sub_fold_1 $4
#python train_rimnet.py fold_5 $1 $2 sub_fold_2 $4
#python train_rimnet.py fold_5 $1 $2 sub_fold_3 $4

#for rimnet
#sh train_2.sh T1 4 rimnet
#export CUDA_VISIBLE_DEVICES=$2
#python train_rimnet.py fold_3 $1 sub_fold_2 $3
#python train_rimnet.py fold_3 $1 sub_fold_3 $3
#python train_rimnet.py fold_4 $1 sub_fold_1 $3
#python train_rimnet.py fold_4 $1 sub_fold_2 $3
#python train_rimnet.py fold_4 $1 sub_fold_3 $3
#python train_rimnet.py fold_5 $1 sub_fold_1 $3
#python train_rimnet.py fold_5 $1 sub_fold_2 $3
#python train_rimnet.py fold_5 $1 sub_fold_3 $3
