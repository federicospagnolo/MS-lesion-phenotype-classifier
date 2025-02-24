#!/bin/bash
#python patch_gen_nested_kfold.py Phase fold_1 train sub_fold_1
python patch_gen_nested_kfold.py Phase fold_1 train sub_fold_2
python patch_gen_nested_kfold.py Phase fold_1 train sub_fold_3
#python patch_gen_nested_kfold.py Phase fold_1 val sub_fold_1
python patch_gen_nested_kfold.py Phase fold_1 val sub_fold_2
python patch_gen_nested_kfold.py Phase fold_1 val sub_fold_3
#python patch_gen_nested_kfold.py Phase fold_1 test
python patch_gen_nested_kfold.py Phase fold_3 train sub_fold_1
python patch_gen_nested_kfold.py Phase fold_3 train sub_fold_2
python patch_gen_nested_kfold.py Phase fold_3 train sub_fold_3
python patch_gen_nested_kfold.py Phase fold_3 val sub_fold_1
python patch_gen_nested_kfold.py Phase fold_3 val sub_fold_2
python patch_gen_nested_kfold.py Phase fold_3 val sub_fold_3
python patch_gen_nested_kfold.py Phase fold_3 test
python patch_gen_nested_kfold.py Phase fold_5 train sub_fold_1
python patch_gen_nested_kfold.py Phase fold_5 train sub_fold_2
python patch_gen_nested_kfold.py Phase fold_5 train sub_fold_3
python patch_gen_nested_kfold.py Phase fold_5 val sub_fold_1
python patch_gen_nested_kfold.py Phase fold_5 val sub_fold_2
python patch_gen_nested_kfold.py Phase fold_5 val sub_fold_3
