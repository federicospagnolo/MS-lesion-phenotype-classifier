#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 00:19:03 2024

@author: federico.spagnolo

usage: python split_data_nPRL.py {contrast_name}

"""

import os
import sys
import pandas as pd
import random
import shutil
from multiprocessing import Pool
from sklearn.model_selection import KFold

contrast = sys.argv[1]

# Set a random seed for reproducibility
random_seed = 42
random.seed(random_seed)

# Base directory containing all patient data
base_dir = "/home/federico.spagnolo/Federico/PRLsConfluent/TS_FS/INsIDER_MS/"

# Directories for different PRL categories
dir_0 = os.path.join(base_dir, f"../{contrast}_0")
dir_1_3 = os.path.join(base_dir, f"../{contrast}_1-3")
dir_4_7 = os.path.join(base_dir, f"../{contrast}_4-7")
dir_above = os.path.join(base_dir, f"../{contrast}_above")

# Directories for 3-fold cross-validation
output_dir = f"/home/federico.spagnolo/Federico/PRLsConfluent/TS_FS/INsIDER_MS_nested_kfold/{contrast}"

# Create PRL-based directories if they don't exist
os.makedirs(dir_0, exist_ok=True)
os.makedirs(dir_1_3, exist_ok=True)
os.makedirs(dir_4_7, exist_ok=True)
os.makedirs(dir_above, exist_ok=True)

# Load the Excel file to get PRL information
xlsx_path = os.path.join(base_dir, "../../INsIDER_prl_stats_corrected.xlsx")
df = pd.read_excel(xlsx_path)

# Extract necessary columns from the Excel file
df = df[['Patient', f'PRL_Count']]

# Function to split patients based on PRLs into respective folders
def split_patients_by_prl(df, base_dir, dir_0, dir_1_3, dir_4_7, dir_above):
    # Get all patient directories
    patient_dirs = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    # Iterate through the patient directories and check PRL number
    for patient in patient_dirs:
        patient_tag = patient.split('_')[1]
        patient_info = df[df['Patient'] == patient_tag]

        if not patient_info.empty:
            prl_number = patient_info[f'PRL_Count'].values[0]  # Get PRL number for the patient
            
            # Determine the destination based on PRL number
            source_dir = os.path.join(base_dir, patient)
            if prl_number == 0:
                shutil.copytree(source_dir, os.path.join(dir_0, patient), dirs_exist_ok=True)
            elif 1 <= prl_number <= 3:
                shutil.copytree(source_dir, os.path.join(dir_1_3, patient), dirs_exist_ok=True)
            elif 4 <= prl_number <= 7:
                shutil.copytree(source_dir, os.path.join(dir_4_7, patient), dirs_exist_ok=True)
            else:
                shutil.copytree(source_dir, os.path.join(dir_above, patient), dirs_exist_ok=True)

# Helper function to copy a single patient's data
def copy_patient_data(args):
    patient, src_dir, dest_dir = args
    src = os.path.join(src_dir, patient)
    dest = os.path.join(dest_dir, patient)
    if not os.path.exists(src):
        print(f"Source directory does not exist: {src}")
        return
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(src, dest, dirs_exist_ok=True)

# Function to distribute patients into train, val, and test sets for 3-fold nested cross-validation
def nested_cross_validation(prl_dir, output_dir, num_folds=5, inner_folds=3):
    # List all patients in the PRL category folder
    patients = [f for f in os.listdir(prl_dir) if os.path.isdir(os.path.join(prl_dir, f))]

    if not patients:
        print(f"No patients found in {prl_dir}")
        return
    
    # Shuffle patients for random sampling
    random.shuffle(patients)
    
    # Outer KFold to split data into `num_folds` (e.g., 3 folds)
    outer_kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
    
    for fold, (train_val_idx, test_idx) in enumerate(outer_kf.split(patients), 1):
        # Create outer fold directory
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        test_dir = os.path.join(fold_dir, "test")
        
        # Create test directory
        os.makedirs(test_dir, exist_ok=True)
        
        # Separate test set (20% of all data per fold)
        test_patients = [patients[i] for i in test_idx]
        
        # Copy patients to test directory
        test_args = [(p, prl_dir, test_dir) for p in test_patients]
        with Pool(6) as pool:
            pool.map(copy_patient_data, test_args)
        
        # Remaining patients for train/val split
        train_val_patients = [patients[i] for i in train_val_idx]
        
        # Inner KFold for train/val split (create sub-folds)
        inner_kf = KFold(n_splits=inner_folds, shuffle=True, random_state=random_seed)
        
        for sub_fold, (train_idx, val_idx) in enumerate(inner_kf.split(train_val_patients), 1):
            # Create sub-fold directories
            sub_fold_dir = os.path.join(fold_dir, f"sub_fold_{sub_fold}")
            train_dir = os.path.join(sub_fold_dir, "train")
            val_dir = os.path.join(sub_fold_dir, "val")
            
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            
            # Get train and validation patients
            train_patients = [train_val_patients[i] for i in train_idx]
            val_patients = [train_val_patients[i] for i in val_idx]
            
            # Prepare arguments for multiprocessing
            train_args = [(p, prl_dir, train_dir) for p in train_patients]
            val_args = [(p, prl_dir, val_dir) for p in val_patients]
            
            # Copy train and validation data
            with Pool(6) as pool:
                pool.map(copy_patient_data, train_args)
                pool.map(copy_patient_data, val_args)
        
        # Print counts for validation
        print(f"Fold {fold} - Test: {len(test_patients)}, Sub-folds with {inner_folds} train/val splits created.")
        
        

# Step 1: Split patients into PRL-based folders
split_patients_by_prl(df, base_dir, dir_0, dir_1_3, dir_4_7, dir_above)

# Step 2: Distribute patients from each PRL folder into 3-fold cross-validation with 5 inner sub-folds
print("Splitting patients with 0 PRLs")
nested_cross_validation(dir_0, output_dir)
print("Splitting patients with 1-3 PRLs")
nested_cross_validation(dir_1_3, output_dir)
print("Splitting patients with 4-7 PRLs")
nested_cross_validation(dir_4_7, output_dir)
print("Splitting patients with more PRLs")
nested_cross_validation(dir_above, output_dir)

print(f"Nested cross-validation completed for contrast {contrast} with 60% train, 20% val, 20% test and 3 sub-folds per outer fold.")
