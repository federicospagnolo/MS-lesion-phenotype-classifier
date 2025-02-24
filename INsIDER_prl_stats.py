##############################
#Written by Federico Spagnolo 
#usage:
#conda activate conflunet
#python INsIDER_prl_stats.py
##############################

import os, csv, sys
import pandas as pd
import shutil
from glob import glob
import nibabel as nib
import numpy as np
from scipy.ndimage import label, generate_binary_structure
from tqdm import tqdm
from multiprocessing import Pool


def compute_prl_stats(data_path, output_excel):
    
    # Extract tags and iterate over the corresponding folders in data_path
    mask_files = glob(data_path + '/*')
    
    # List to hold all results
    all_results = []
    
    # Use multiprocessing Pool to parallelize the processing
    with Pool(processes=6) as pool:
        # Map the process_folder function to each data_folder_path
        results = pool.map(process_folder, mask_files)
        all_results.extend(results)
        
    # Filter out None results (in case of any errors)
    all_results = [res for res in all_results if res is not None]
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Write to Excel
    df.to_excel(output_excel, index=False)    

def process_folder(mask_file):
        
        prl_volume = []
        hyper_volume = []
        iso_volume = []
        hypo_volume = []
        prl_count = []
        hyper_count = []
        iso_count = []
        hypo_count = []
        
        s = generate_binary_structure(3,1)
       
        if os.path.isfile(mask_file):
            basename = mask_file.split('/')[-1]
            patient = basename.split('_')[0]
            session = basename.split('_')[1]
            print(f'Processing {patient}/{session}')
   
            # Compute stats
            try:
                mask = nib.load(mask_file).get_fdata()
                #print("File loaded successfully!")
            except Exception as e:
                print(f"An error occurred with {patient}/{session}: {e}")
            affine = nib.load(mask_file).affine
            
            # Delete annotation helpers
            #if np.any(mask[:,:,0]) != 0:
                #print(f'{patient}/{session}')
            #    mask[:,:,0] = 0
            #    nib.save(nib.Nifti1Image(mask, affine), mask_file)
            #    mask = nib.load(mask_file).get_fdata()
                
            voxel_dimensions = nib.affines.voxel_sizes(affine)
            
            conversion_factor = voxel_dimensions[0] * voxel_dimensions[1] * voxel_dimensions[2] / 1000

            prl_clusters, _ = label(mask == 1, structure=s)
            for lesion in range (1, np.max(_) + 1):
                lesion_volume = np.sum(prl_clusters == lesion)
                if lesion_volume < 5:
                    coordinates = np.where(prl_clusters == lesion)
                    x, y, z = coordinates[0][0], coordinates[1][0], coordinates[2][0]
                    
                    
                    with open("info.txt", "a") as file:
                         print("PRL < 5 voxels in ", patient, session, x+1, y+1, z+1, file=file)
            count_temp = np.max(prl_clusters)
            
            prl_clusters, _ = label(mask == 7, structure=s)
            for lesion in range (1, np.max(_) + 1):
                lesion_volume = np.sum(prl_clusters == lesion)
                if lesion_volume < 5:
                    coordinates = np.where(prl_clusters == lesion)
                    x, y, z = coordinates[0][0], coordinates[1][0], coordinates[2][0]
                    
                    with open("info.txt", "a") as file:
                         print("PRL < 5 voxels in ", patient, session, x+1, y+1, z+1, file=file)
            prl_count.append(np.max(prl_clusters) + count_temp)
            
            hyper_clusters, _ = label(mask == 2, structure=s)
            for lesion in range (1, np.max(_) + 1):
                lesion_volume = np.sum(hyper_clusters == lesion)
                if lesion_volume < 5:
                    coordinates = np.where(hyper_clusters == lesion)
                    x, y, z = coordinates[0][0], coordinates[1][0], coordinates[2][0]
                    
                    with open("info.txt", "a") as file:
                         print("HYPER < 5 voxels in ", patient, session, x+1, y+1, z+1, file=file)
            count_temp = np.max(hyper_clusters)
            
            hyper_clusters, _ = label(mask == 8, structure=s)
            for lesion in range (1, np.max(_) + 1):
                lesion_volume = np.sum(hyper_clusters == lesion)
                if lesion_volume < 5:
                    coordinates = np.where(hyper_clusters == lesion)
                    x, y, z = coordinates[0][0], coordinates[1][0], coordinates[2][0]
                    
                    with open("info.txt", "a") as file:
                         print("HYPER < 5 voxels in ", patient, session, x+1, y+1, z+1, file=file)
            hyper_count.append(np.max(hyper_clusters) + count_temp)
            
            iso_clusters, _ = label(mask == 3, structure=s)
            for lesion in range (1, np.max(_) + 1):
                lesion_volume = np.sum(iso_clusters == lesion)
                if lesion_volume < 5:
                    coordinates = np.where(iso_clusters == lesion)
                    x, y, z = coordinates[0][0], coordinates[1][0], coordinates[2][0]
                    
                    with open("info.txt", "a") as file:
                         print("ISO < 5 voxels in ", patient, session, x+1, y+1, z+1, file=file)
            count_temp = np.max(iso_clusters)
            
            iso_clusters, _ = label(mask == 9, structure=s)
            for lesion in range (1, np.max(_) + 1):
                lesion_volume = np.sum(iso_clusters == lesion)
                if lesion_volume < 5:
                    coordinates = np.where(iso_clusters == lesion)
                    x, y, z = coordinates[0][0], coordinates[1][0], coordinates[2][0]
                    
                    with open("info.txt", "a") as file:
                         print("ISO < 5 voxels in ", patient, session, x+1, y+1, z+1, file=file)
            iso_count.append(np.max(iso_clusters) + count_temp)
            
            hypo_clusters, _ = label(mask == 4, structure=s)
            for lesion in range (1, np.max(_) + 1):
                lesion_volume = np.sum(hypo_clusters == lesion)
                if lesion_volume < 5:
                    coordinates = np.where(hypo_clusters == lesion)
                    x, y, z = coordinates[0][0], coordinates[1][0], coordinates[2][0]
                    
                    with open("info.txt", "a") as file:
                         print("HYPO < 5 voxels in ", patient, session, x+1, y+1, z+1, file=file)
            hypo_count.append(np.max(hypo_clusters))          
                       
            prl_volume.append(np.sum(mask == 1) + np.sum(mask == 7))
            hyper_volume.append(np.sum(mask == 2) + np.sum(mask == 8))
            iso_volume.append(np.sum(mask == 3) + np.sum(mask == 9))
            hypo_volume.append(np.sum(mask == 4))
              
            # Return the results as a dictionary
            return {
                'Patient': patient,
                'Session': session,
                'PRL_Count': int(prl_count[0]),
                'HYPER_Count': int(hyper_count[0]),
                'ISO_Count': int(iso_count[0]),
                'HYPO_Count': int(hypo_count[0]),
                'PRL_Volume (ml)': (int(prl_volume[0]) / int(prl_count[0]) * conversion_factor) if not  int(prl_count[0]) == 0 else 0,
                'HYPER_Volume (ml)': (int(hyper_volume[0]) / int(hyper_count[0]) * conversion_factor) if not int(hyper_count[0] == 0) else 0,
                'ISO_Volume (ml)': (int(iso_volume[0]) / int(iso_count[0]) * conversion_factor) if not int(iso_count[0]) == 0 else 0,
                'HYPO_Volume (ml)': (int(hypo_volume[0]) / int(hypo_count[0]) * conversion_factor) if not int(hypo_count[0]) == 0 else 0
                 }
            
            
            
# Define the paths
data_path = '/home/federico.spagnolo/Federico/PRLsConfluent/Corrected_masks_nosquares'

output_excel = 'INsIDER_prl_stats_corrected.xlsx'  # Output Excel file path
os.system('touch info.txt')

# Run the function
compute_prl_stats(data_path, output_excel)
