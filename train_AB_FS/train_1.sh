#!/bin/bash
#usage: sh train_1.sh T1 Phase Mask 5
export CUDA_VISIBLE_DEVICES=$4
python train_rimnet.py fold_1 $1 $2 $3 sub_fold_1
python train_rimnet.py fold_1 $1 $2 $3 sub_fold_2
python train_rimnet.py fold_1 $1 $2 $3 sub_fold_3
python train_rimnet.py fold_2 $1 $2 $3 sub_fold_1
python train_rimnet.py fold_2 $1 $2 $3 sub_fold_2
python train_rimnet.py fold_2 $1 $2 $3 sub_fold_3
python train_rimnet.py fold_3 $1 $2 $3 sub_fold_1
