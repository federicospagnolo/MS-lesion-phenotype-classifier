#!/bin/bash
export CUDA_VISIBLE_DEVICES=$4
python train_rimnet.py fold_3 $1 $2 $3 sub_fold_2
python train_rimnet.py fold_3 $1 $2 $3 sub_fold_3
python train_rimnet.py fold_4 $1 $2 $3 sub_fold_1
python train_rimnet.py fold_4 $1 $2 $3 sub_fold_2
python train_rimnet.py fold_4 $1 $2 $3 sub_fold_3
python train_rimnet.py fold_5 $1 $2 $3 sub_fold_1
python train_rimnet.py fold_5 $1 $2 $3 sub_fold_2
python train_rimnet.py fold_5 $1 $2 $3 sub_fold_3
