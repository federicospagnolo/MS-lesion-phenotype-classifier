#!/bin/bash
#usage: sh test_2.sh T1 QSM Mask 4 multiclass
export CUDA_VISIBLE_DEVICES=$4
python test_rimnet.py --modalities $1 $2 $3 --folder 1 --sub_folder 1 --task $5
python test_rimnet.py --modalities $1 $2 $3 --folder 2 --sub_folder 2 --task $5
python test_rimnet.py --modalities $1 $2 $3 --folder 3 --sub_folder 3 --task $5
python test_rimnet.py --modalities $1 $2 $3 --folder 4 --sub_folder 2 --task $5
python test_rimnet.py --modalities $1 $2 $3 --folder 5 --sub_folder 1 --task $5
