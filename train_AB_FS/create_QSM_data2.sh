#!/bin/bash
python patch_gen_nested_kfold.py QSM fold_2 train sub_fold_1 binary
python patch_gen_nested_kfold.py QSM fold_2 train sub_fold_2 binary
python patch_gen_nested_kfold.py QSM fold_2 train sub_fold_3 binary
python patch_gen_nested_kfold.py QSM fold_2 val sub_fold_1 binary
python patch_gen_nested_kfold.py QSM fold_2 val sub_fold_2 binary
python patch_gen_nested_kfold.py QSM fold_2 val sub_fold_3 binary
python patch_gen_nested_kfold.py QSM fold_2 test binary
python patch_gen_nested_kfold.py QSM fold_4 train sub_fold_1 binary
python patch_gen_nested_kfold.py QSM fold_4 train sub_fold_2 binary
python patch_gen_nested_kfold.py QSM fold_4 train sub_fold_3 binary
python patch_gen_nested_kfold.py QSM fold_4 val sub_fold_1 binary
python patch_gen_nested_kfold.py QSM fold_4 val sub_fold_2 binary
python patch_gen_nested_kfold.py QSM fold_4 val sub_fold_3 binary
python patch_gen_nested_kfold.py QSM fold_4 test binary
python patch_gen_nested_kfold.py QSM fold_5 test binary
