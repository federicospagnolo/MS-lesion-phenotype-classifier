#!/bin/bash
#usage: sh test_1.sh T1 Phase Mask

python test_rimnet.py --modalities $1 $2 $3 --folder 1 --sub_folder 2
python test_rimnet.py --modalities $1 $2 $3 --folder 2 --sub_folder 1
python test_rimnet.py --modalities $1 $2 $3 --folder 3 --sub_folder 3
python test_rimnet.py --modalities $1 $2 $3 --folder 4 --sub_folder 1
python test_rimnet.py --modalities $1 $2 $3 --folder 5 --sub_folder 3
