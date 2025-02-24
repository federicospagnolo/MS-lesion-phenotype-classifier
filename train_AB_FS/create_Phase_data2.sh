#!/bin/bash
python patch_gen_nested_kfold.py Phase fold_2 train sub_fold_1
python patch_gen_nested_kfold.py Phase fold_2 train sub_fold_2
python patch_gen_nested_kfold.py Phase fold_2 train sub_fold_3
python patch_gen_nested_kfold.py Phase fold_2 val sub_fold_1
python patch_gen_nested_kfold.py Phase fold_2 val sub_fold_2
python patch_gen_nested_kfold.py Phase fold_2 val sub_fold_3
python patch_gen_nested_kfold.py Phase fold_2 test
python patch_gen_nested_kfold.py Phase fold_4 train sub_fold_1
python patch_gen_nested_kfold.py Phase fold_4 train sub_fold_2
python patch_gen_nested_kfold.py Phase fold_4 train sub_fold_3
python patch_gen_nested_kfold.py Phase fold_4 val sub_fold_1
python patch_gen_nested_kfold.py Phase fold_4 val sub_fold_2
python patch_gen_nested_kfold.py Phase fold_4 val sub_fold_3
python patch_gen_nested_kfold.py Phase fold_4 test
python patch_gen_nested_kfold.py Phase fold_5 test
