#!/bin/bash
python patch_gen_nested_kfold.py QSM fold_1 train sub_fold_1 binary
python patch_gen_nested_kfold.py QSM fold_1 train sub_fold_2 binary
python patch_gen_nested_kfold.py QSM fold_1 train sub_fold_3 binary
python patch_gen_nested_kfold.py QSM fold_1 val sub_fold_1 binary
python patch_gen_nested_kfold.py QSM fold_1 val sub_fold_2 binary
python patch_gen_nested_kfold.py QSM fold_1 val sub_fold_3 binary
python patch_gen_nested_kfold.py QSM fold_1 test binary
python patch_gen_nested_kfold.py QSM fold_3 train sub_fold_1 binary
python patch_gen_nested_kfold.py QSM fold_3 train sub_fold_2 binary
python patch_gen_nested_kfold.py QSM fold_3 train sub_fold_3 binary
python patch_gen_nested_kfold.py QSM fold_3 val sub_fold_1 binary
python patch_gen_nested_kfold.py QSM fold_3 val sub_fold_2 binary
python patch_gen_nested_kfold.py QSM fold_3 val sub_fold_3 binary
python patch_gen_nested_kfold.py QSM fold_3 test binary
python patch_gen_nested_kfold.py QSM fold_5 train sub_fold_1 binary
python patch_gen_nested_kfold.py QSM fold_5 train sub_fold_2 binary
python patch_gen_nested_kfold.py QSM fold_5 train sub_fold_3 binary
python patch_gen_nested_kfold.py QSM fold_5 val sub_fold_1 binary
python patch_gen_nested_kfold.py QSM fold_5 val sub_fold_2 binary
python patch_gen_nested_kfold.py QSM fold_5 val sub_fold_3 binary
