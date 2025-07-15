#!/bin/bash
#usage: sh train_1.sh T1 Phase Mask 5 multiclass
export CUDA_VISIBLE_DEVICES=$4
python train_rimnet.py fold_1 $1 $2 $3 sub_fold_1 $5
python train_rimnet.py fold_1 $1 $2 $3 sub_fold_2 $5
python train_rimnet.py fold_1 $1 $2 $3 sub_fold_3 $5
python train_rimnet.py fold_2 $1 $2 $3 sub_fold_1 $5
python train_rimnet.py fold_2 $1 $2 $3 sub_fold_2 $5
python train_rimnet.py fold_2 $1 $2 $3 sub_fold_3 $5
python train_rimnet.py fold_3 $1 $2 $3 sub_fold_1 $5

#for rimnet
#sh train_1.sh T1 QSM 4 rimnet
#export CUDA_VISIBLE_DEVICES=$3
#python train_rimnet.py fold_1 $1 $2 sub_fold_1 $4
#python train_rimnet.py fold_1 $1 $2 sub_fold_2 $4
#python train_rimnet.py fold_1 $1 $2 sub_fold_3 $4
#python train_rimnet.py fold_2 $1 $2 sub_fold_1 $4
#python train_rimnet.py fold_2 $1 $2 sub_fold_2 $4
#python train_rimnet.py fold_2 $1 $2 sub_fold_3 $4
#python train_rimnet.py fold_3 $1 $2 sub_fold_1 $4

#for rimnetmono
#sh train_1.sh T1 4 rimnetmono
#export CUDA_VISIBLE_DEVICES=$2
#python train_rimnet.py fold_1 $1 sub_fold_1 $3
#python train_rimnet.py fold_1 $1 sub_fold_2 $3
#python train_rimnet.py fold_1 $1 sub_fold_3 $3
#python train_rimnet.py fold_2 $1 sub_fold_1 $3
#python train_rimnet.py fold_2 $1 sub_fold_2 $3
#python train_rimnet.py fold_2 $1 sub_fold_3 $3
#python train_rimnet.py fold_3 $1 sub_fold_1 $3
