#!/bin/bash
#usage: sh test_1.sh T1 Phase/QSM Mask 4 binary
export CUDA_VISIBLE_DEVICES=$4
python test_rimnet.py --modalities $1 $2 $3 --folder 1 --sub_folder 1 --task $5
python test_rimnet.py --modalities $1 $2 $3 --folder 2 --sub_folder 2 --task $5
python test_rimnet.py --modalities $1 $2 $3 --folder 3 --sub_folder 3 --task $5
python test_rimnet.py --modalities $1 $2 $3 --folder 4 --sub_folder 2 --task $5
python test_rimnet.py --modalities $1 $2 $3 --folder 5 --sub_folder 3 --task $5

#2 modalities
#export CUDA_VISIBLE_DEVICES=$3
#python test_rimnet.py --modalities $1 $2 --folder 1 --sub_folder 1 --task $4
#python test_rimnet.py --modalities $1 $2 --folder 2 --sub_folder 2 --task $4
#python test_rimnet.py --modalities $1 $2 --folder 3 --sub_folder 2 --task $4
#python test_rimnet.py --modalities $1 $2 --folder 4 --sub_folder 2 --task $4
#python test_rimnet.py --modalities $1 $2 --folder 5 --sub_folder 1 --task $4

#1 modality
#export CUDA_VISIBLE_DEVICES=$2
#python test_rimnet.py --modalities $1 --folder 1 --sub_folder 3 --task $3
#python test_rimnet.py --modalities $1 --folder 2 --sub_folder 3 --task $3
#python test_rimnet.py --modalities $1 --folder 3 --sub_folder 1 --task $3
#python test_rimnet.py --modalities $1 --folder 4 --sub_folder 2 --task $3
#python test_rimnet.py --modalities $1 --folder 5 --sub_folder 1 --task $3
