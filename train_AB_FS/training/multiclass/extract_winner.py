# Usage: python extract_winner.py T1 Phase QSM
import os, sys
import pandas as pd
import glob

cont_1 = sys.argv[1]
cont_2 = sys.argv[2]
if len(sys.argv) > 3:
    cont_3 = sys.argv[3]
    modalities = [f'{cont_2}_{cont_1}_Mask', f'{cont_3}_{cont_1}_Mask']
else:
    # Modalities to be processed
    modalities = [f'{cont_2}_{cont_1}_Mask']


# Base directory where the folds are located
base_dir = os.path.dirname(os.path.realpath(__file__))

# List of folds
folds = [f"fold_{i}" for i in range(1, 6)]

# Result dictionary to store the highest sub_fold and F1 for each fold and modality
results = []

def find_max_f1_in_fold(fold_path, modality):
    """This function finds the sub_fold with the maximum F1 score for the given fold and modality."""
    max_f1 = -float('inf')
    max_iso = -float('inf')
    max_prl = -float('inf')
    max_hyper = -float('inf')
    best_sub_fold = None
    
    # Loop over all sub-folders in the current fold directory
    for sub_fold in os.listdir(fold_path):
        sub_fold_path = os.path.join(fold_path, sub_fold, modality)

        if os.path.isdir(sub_fold_path):  # Only process sub-folders
            # Use glob to find the wildcard directory (e.g., exp*)
            wildcard_dirs = glob.glob(os.path.join(sub_fold_path, "exp*"))
            
            if not wildcard_dirs:  # If the wildcard directory not exists
                print("wildcard directory not exists")
                break    
            summary_file = os.path.join(wildcard_dirs[0], "summary.csv")
            
            if os.path.exists(summary_file):
                # Read the CSV file
                #df = pd.read_csv(summary_file, delimiter="\t", dtype=str)  # No predefined header
                f1_iso = []
                f1_prl = []
                f1_hyper = []
                f1_values = []
                with open(summary_file, "r") as f:
                  header = f.readline().strip().split(",")  # Read the header and split by comma
                  f1_index = header.index("F1_per_class")
                  lines = f.readlines()
                  
                  for line in lines:
                     values = line.strip().split(",")
                     f1_iso.append(float(values[f1_index].strip('"').strip('[]')))
                     f1_prl.append(float(values[f1_index + 1].strip('"').strip('[]')))
                     f1_hyper.append(float(values[f1_index + 2].strip('"').strip('[]')))

                # Get the maximum F1 value in this sub-folder
                for iso, prl, hyper in zip(f1_iso, f1_prl, f1_hyper):
                    f1_values.append((iso + prl + hyper) / 3)
                
                idx = f1_values.index(max(f1_values))
                max_f1_in_sub_fold = f1_values[idx]
                max_iso_in_sub_fold = f1_iso[idx]
                max_prl_in_sub_fold = f1_prl[idx]
                max_hyper_in_sub_fold = f1_hyper[idx]
                
                # Update the max F1 and the corresponding sub-fold if the new one is larger
                if max_f1_in_sub_fold > max_f1:
                    max_f1 = max_f1_in_sub_fold
                    max_iso = max_iso_in_sub_fold
                    max_prl = max_prl_in_sub_fold
                    max_hyper = max_hyper_in_sub_fold
                    best_sub_fold = sub_fold
    
    return best_sub_fold, max_f1, max_iso, max_prl, max_hyper

# Iterate through each modality
for modality in modalities:
    # Process each fold
    for fold in folds:
        fold_path = os.path.join(base_dir, fold)
        
        # Find the sub_fold with the highest F1 score
        best_sub_fold, best_f1, best_iso, best_prl, best_hyper = find_max_f1_in_fold(fold_path, modality)
        
        # Store the result in the results list
        results.append({
            "Fold": fold,
            "Modality": modality,
            "Best_Sub_Fold": best_sub_fold,
            "Best_F1": best_f1,
            "Best_F1_iso": best_iso,
            "Best_F1_PRL": best_prl,
            "Best_F1_hyper": best_hyper
        })

# Convert the results into a DataFrame
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv("highest_f1_results.csv", index=False)

print("Results saved to highest_f1_results.csv")
