#usage: python test_summary.py QSM

import os, sys
import glob
import csv
import re, ast
import pandas as pd
import numpy as np

def process_folds(base_path, output_csv, label):
    """
    Function to process the folds and calculate mean and std for Phase or QSM.
    """
    all_data = []
    folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']

    for fold in folds:
        # Find all summary.csv files in the current fold
        csv_path_pattern = os.path.join(base_path, fold, '*', 'summary.csv')
        csv_files = glob.glob(csv_path_pattern)
        
        for csv_file in csv_files:
            # Read the summary.csv file
            with open(csv_file, "r") as f:
                  reader = csv.reader(f, delimiter="\n")
                  header = next(reader)[0].split(',')  # Read the header and split by comma
                  values = next(reader)[0].split('],')
                  chunk0 = values[0].replace('[','')
                  chunk1 = values[1].split(",[")[0]
                  chunk2 = values[1].split(",[")[1]
                  chunk3 = values[2]
                  
                  values = [chunk0.split(',')] + chunk1.split(',') + [chunk2.split(',')] + chunk3.split(',')
                  f1xclass_index = header.index('F1_per_class')
                  f1_index = header.index('F1')
                  f1w_index = header.index('F1 weighted')
                  prec_index = header.index('Precision macro')
                  rec_index = header.index('Recall macro')
                  precw_index = header.index('Precision weighted')
                  recw_index = header.index('Recall weighted')
                  auc_index = header.index('AUC')
                  
                  f1xiso = float(values[f1xclass_index][0])
                  f1xprl = float(values[f1xclass_index][1])
                  f1xhyp = float(values[f1xclass_index][2])
                  f1 = float(values[f1_index])
                  f1w = float(values[f1w_index])
                  prec = float(values[prec_index])
                  rec = float(values[rec_index])
                  precw = float(values[precw_index])
                  recw = float(values[recw_index])
                  auc = values[auc_index][0].strip().split(' ')
                  auc = list(map(float, auc))
                  auc_macro = np.mean(auc)
                  auc_prl = auc[1]
                  
            
            # Store the data in a structured format
            all_data.append({"fold": fold, "type": label, "F1 iso": f1xiso, "F1 prl": f1xprl, "F1 hyp": f1xhyp, "F1": f1, "F1 weighted": f1w,
            "Precision macro": prec, "Recall macro": rec, "Precision weighted": precw, "Recall weighted": recw, "AUC": auc_macro, "AUC PRL": auc_prl})
    
    df = pd.DataFrame(all_data)
    print(df)

    # Calculate mean and std for each column (excluding fold and type columns)
    mean_df = df.mean(numeric_only=True).to_frame().T
    mean_df['fold'] = 'MEAN'
    mean_df['type'] = label
    
    std_df = df.std(numeric_only=True).to_frame().T
    std_df['fold'] = 'STD'
    std_df['type'] = label
    
    # Append mean and std to the combined dataframe
    combined_df = pd.concat([df, mean_df, std_df], ignore_index=True)
    
    # If the CSV already exists, append the data; otherwise, create a new file
    if os.path.exists(output_csv):
        combined_df.to_csv(output_csv, mode='a', header=True, index=False)
    else:
        combined_df.to_csv(output_csv, header=True, index=False)

cont = sys.argv[1]

# Paths for Phase and QSM
ph_path = f'{cont}_T1_nested_kfold'

# Output file (same for both Phase and QSM)
output_csv = 'combined_summary.csv'
os.system(f"touch {output_csv}")

# Process Phase and QSM data and write to the same CSV
print("Processing QSM data...")
process_folds(ph_path, output_csv, label=cont)

