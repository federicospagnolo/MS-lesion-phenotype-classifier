"""
Created on Fri Jun 28 15:51:31 2024

@author: Federico Spagnolo
"""

import glob, os, sys
import csv
from tqdm import tqdm
import nibabel as nib
import scipy.ndimage as ndimage
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import ListedColormap

def process_patch(args):
    flair, t1, phase, qsm, label_map, affine, patient, session, start, end, idx, com = args
    
    #affine_mirrored = affine.copy()
    # Flip the sign of the X scaling factor (first element of the first column)
    #affine_mirrored[0, 0] = -affine_mirrored[0, 0]
    #mirrored_mask = np.flip(label_map, axis=0)
    #mirrored_nii = nib.Nifti1Image(mirrored_mask, affine_mirrored)
    #nib.save(mirrored_nii, f'{output_path}{patient}_{session}_{idx}_MASK_with_contralateral.nii.gz')
    
    # Crop the patch from input images
    mask = label_map[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    mask[mask>0] = 1
    
    # Dilate
    dilated_mask = ndimage.binary_dilation(mask, structure=struct, iterations=1).astype(float)
    
    flair_patch = flair[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    t1_patch = t1[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    phase_patch = phase[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    qsm_patch = qsm[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    
    t1_patch = (t1_patch - t1_patch.min()) / (t1_patch.max() - t1_patch.min()) * (1.0 - 0.0) + 0.0
    flair_patch = (flair_patch - flair_patch.min()) / (flair_patch.max() - flair_patch.min()) * (1.0 - 0.0) + 0.0
    phase_patch = (phase_patch - phase_patch.min()) / (phase_patch.max() - phase_patch.min()) * (1.0 - 0.0) + 0.0
    qsm_patch = (qsm_patch - (-200)) / (200 - (-200)) * (1.0 - 0.0) + 0.0
    #qsm_patch = (qsm_patch - qsm_patch.min()) / (qsm_patch.max() - qsm_patch.min()) * (1.0 - 0.0) + 0.0
    
    flair_img = nib.Nifti1Image(flair_patch, affine)
    t1_img = nib.Nifti1Image(t1_patch, affine)
    phase_img = nib.Nifti1Image(phase_patch, affine)
    qsm_img = nib.Nifti1Image(qsm_patch, affine)
    mask_img = nib.Nifti1Image(dilated_mask, affine)
    nib.save(flair_img, f'{output_path}{patient}_{session}_{idx}_FLAIR.nii.gz')
    nib.save(t1_img, f'{output_path}{patient}_{session}_{idx}_T1.nii.gz')
    nib.save(phase_img, f'{output_path}{patient}_{session}_{idx}_Phase.nii.gz')
    nib.save(qsm_img, f'{output_path}{patient}_{session}_{idx}_QSM.nii.gz')
    nib.save(mask_img, f'{output_path}{patient}_{session}_{idx}_MASK.nii.gz')
    
    #if (int(idx) >= 1000 and int(idx) < 2000) or (int(idx) >= 7000 and int(idx) < 8000):
    #    label = '1'
    #else:
    #    label = '0'
    
    label = int(idx[0])
    
    # Open the CSV file in append mode to add rows
    row = [patient+'_'+session, idx, f'{patient}_{session}_{idx}_FLAIR.nii.gz', f'{patient}_{session}_{idx}_T1.nii.gz', f'{patient}_{session}_{idx}_QSM.nii.gz', f'{patient}_{session}_{idx}_Phase.nii.gz', f'{patient}_{session}_{idx}_MASK.nii.gz', label, str(com)]
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the rows one by one

        writer.writerow(row)


def process_labels(mask, flair, t1, phase, qsm, affine, patient, session, output_file):
    
    subtypes = [1,2,3,4,7,8,9] # labels in QSM assessment
    
    for subtype in subtypes:

      the_subtype = (mask == subtype).astype(np.uint8)
      if np.sum(the_subtype) == 0:
        continue
      label_map, labels = ndimage.label(the_subtype, structure=struct)
      #mask_img = nib.Nifti1Image(label_map, affine)
      #nib.save(mask_img, f'{output_path}{patient}_{session}_labelmap_{subtype}.nii.gz')
      
      # Compute centers of mass for all labels at once
      coms = ndimage.center_of_mass(the_subtype, labels=label_map, index=range(1, labels + 1))

      args_list = []

      for label, com in zip(range(1, labels + 1), coms):

        the_cluster = (label_map == label).astype(np.uint8)
        #mask_img = nib.Nifti1Image(the_cluster, affine)
        #nib.save(mask_img, f'{output_path}{patient}_{session}_the_cluster_{subtype}_{label}.nii.gz')
        com = (int(com[0]), int(com[1]), int(com[2]))

        # Calculate the patch bounds, ensuring they stay within image bounds  
        half_patch_size = tuple(size // 2 for size in PATCH_SIZE)
        start = [max(0, com[i] - half_patch_size[i]) for i in range(3)]
        end = [min(flair.shape[i], com[i] + half_patch_size[i]) for i in range(3)]

        if label > 100:
            idx = f'{subtype}{label-1}'
        elif label > 10:
            idx = f'{subtype}0{label-1}'
        else:
            idx = f'{subtype}00{label-1}'

        args_list.append((flair, t1, phase, qsm, the_cluster, affine, patient, session, start, end, idx, com))
        
        if str(folder_set) == 'train' and task == 'binary' and subtype in {1, 7}:
            # For training set generate additional patches
            offsets = [(5, 0, 0), (0, 5, 0), (0, 0, 5), (5, 5, 0), (0, 5, 5), (5, 0, 5), (5, 5, 5)]
            
            for i, offset in enumerate(offsets, start=1):
                new_com = tuple(com[i] + offset[i] for i in range(3))
                new_start = [max(0, new_com[i] - half_patch_size[i]) for i in range(3)]
                new_end = [min(flair.shape[i], new_com[i] + half_patch_size[i]) for i in range(3)]
                new_idx = str(int(idx) + 50 * i)  # Generating new index based on offset
                args_list.append((flair, t1, phase, qsm, the_cluster, affine, patient, session, new_start, new_end, new_idx, new_com))
            
        elif str(folder_set) == 'train' and task == 'multiclass' and subtype in {1, 2, 7, 8}:  
            if subtype in {1, 7}:
                offsets = [(5, 0, 0), (0, 5, 0), (0, 0, 5), (5, 5, 0), (0, 5, 5), (5, 0, 5), (5, 5, 5)]
            elif subtype in {2, 8}:
                offsets = [(5, 0, 0), (0, 5, 0), (0, 0, 5), (5, 5, 0)]      
                
            for i, offset in enumerate(offsets, start=1):
                new_com = tuple(com[i] + offset[i] for i in range(3))
                new_start = [max(0, new_com[i] - half_patch_size[i]) for i in range(3)]
                new_end = [min(flair.shape[i], new_com[i] + half_patch_size[i]) for i in range(3)]
                new_idx = str(int(idx) + 50 * i)  # Generating new index based on offset
                args_list.append((flair, t1, phase, qsm, the_cluster, affine, patient, session, new_start, new_end, new_idx, new_com))
  
      # Process patches in parallel
      with Pool(8) as pool:
         pool.map(process_patch, args_list)

###################################
ph_contrast = sys.argv[1] # choose between Phase and QSM
folder = sys.argv[2]  # choose between fold_1 --- fold_5
folder_set = sys.argv[3] # choose between train, val and test
if str(folder_set) == 'test':
     sub_folder = ''
else:     
     sub_folder = sys.argv[4]  # choose between sub_fold_1 --- sub_fold_3
###################################

folders = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
sub_folders = ['sub_fold_1', 'sub_fold_2', 'sub_fold_3']

task = "binary"
#task = "multiclass"

PATCH_SIZE = (28, 28, 28)
if str(folder_set) != 'test':
      input_path = f'/home/federico.spagnolo/Federico/PRLsConfluent/TS_FS/INsIDER_MS_nested_kfold/{ph_contrast}/{folder}/{sub_folder}/{folder_set}'
      output_path = f'/home/federico.spagnolo/Federico/PRLsConfluent/train_AB_FS/INsIDER_MS_patches_nested_kfold_{task}/{folder}/{sub_folder}/{folder_set}/'
else:
      input_path = f'/home/federico.spagnolo/Federico/PRLsConfluent/TS_FS/INsIDER_MS_nested_kfold/{ph_contrast}/{folder}/{folder_set}'         
      output_path = f'/home/federico.spagnolo/Federico/PRLsConfluent/train_AB_FS/INsIDER_MS_patches_nested_kfold_{task}/{folder}/{folder_set}/'

output_file = output_path + f'{folder_set}_data.csv'
struct = np.array(ndimage.generate_binary_structure(3, 1))

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Masks filenames
filenames = glob.glob(input_path + f'/*/*/PRL_mask.nii.gz')

# CSV header
column_names = ["sub_id", "patch_id", f"cont_FLAIR", f"cont_T1", f"cont_QSM", f"cont_Phase", f"{ph_contrast}_Mask", "Label", "extra_info"]
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(column_names)

print(f"Building patches...")
for filename in tqdm(filenames):

    path = os.path.dirname(filename)
    patient = path.split('/')[-2]
    session = path.split('/')[-1]
    
    # Skip if FLAIR patch 0 exists
    #if os.path.exists(f'{output_path}{patient}_{session}_2000_FLAIR.nii.gz'):
    #    continue
    
    #print(f"Processing patient {patient}, session {session}...")
    flair_name = os.path.join(path, "FLAIR_in_EPI.nii.gz")
    mask_name = os.path.join(path, f"PRL_mask.nii.gz")
    t1_name = os.path.join(path, "MP2RAGE_UNI_masked_in_EPI.nii.gz")
    qsm_name = os.path.join(path, "QSM.nii.gz")
    phase_name = os.path.join(path, "Phase_bet.nii.gz")
    
    affine = nib.load(flair_name).affine
    mask = nib.load(mask_name).get_fdata()
    phase = nib.load(phase_name).get_fdata()
    qsm = nib.load(qsm_name).get_fdata()
    flair = nib.load(flair_name).get_fdata()
    t1 = nib.load(t1_name).get_fdata()
    
    # QSM clipping if not done earlier
    #if ph_contrast == 'QSM':
    #    ph = np.clip(ph, -200, 200)
            
    # Pipeline for masks 
    process_labels(mask, flair, t1, phase, qsm, affine, patient, session, output_file)
