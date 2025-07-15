<h1 align="center">MS lesion phenotype classifier </h1>

> Scripts to perform multi-class classification of MS lesions based on their phenotype.<br /> Lesion phenotype classifier can use quantitative susceptibility mapping (QSM) and T1-weighted MR images to predict rim positive, hyper-intense, and iso-/hypo-intense lesions. In addition, the same classifier can be used to distinguish PRL positive and negative lesions using either QSM or filtered phase unwrapped (PU) and T1-weighted images.

## 🚀 Usage

First, make sure you have python >=3.8 installed.

To build the environment, an installation of conda or miniconda is needed. Once it is installed, please use
```sh
conda env create -f environment.yml
```
to create the tested environment using the provided `environment.yml` file. Then activate the environment with
```sh
conda activate lpc
```

The general structure of the directories used in this study is the following:
- The folders "Corrected_masks_nosquares" and "Corrected_masks_nosquares_phase" respectively contain the multiclass annotated masks based on QSM and the binary annotated masks based on PU. The file "files.txt" provides an example of filenames used in this study. 
- The folder "TS_FS" contains the dataset named INsIDER_MS. Inside, patients are named "INsIDER_P001", "INsIDER_P002", etc... Each patient has a session (visit) folder, in our case named "20190123". Each session contains nifti image files specified in the text file "files.txt", which correspond to the FLAIR, MP2RAGE, PU, the multiclass annotation (for QSM), the binary annotation (for PU), and QSM.
- The folder "train_AB_FS" contains key files for the generation of patches, training and testing of networks.

First, move to the project folder and obtain an .xlsx file with the demographic information of MS lesions by running
```sh
python INsIDER_prl_stats.py
```
editing the input and output data paths to obtain information either on QSM-based or on PU-based annotations.

Then, perform patients stratification based on the number of PRLs by running
```sh
python split_data_nPRL.py {contrast_name}
```
editing the variable "xlsx_path" to match the .xlsx generated at the previous step. This will also split the dataset following a nested cross-validation approach, creating a data structure in the fold "INsIDER_MS_patches_nested_kfold_binary" (for the binary masks example). Editing the options "num_folds" and "inner_folds" in the function "nested_cross_validation" will modify the number of outer test folds and train/val splits, respectively.

Move to the directory "train_AB_FS", where four bash files can be used to generate patches into the directory "INsIDER_MS_patches_nested_kfold_{task}", where task can be either binary or multiclass. If you want to create patches for both QSM-based data and PU, run
```sh
sh create_QSM_data1.sh
sh create_QSM_data2.sh
sh create_Phase_data1.sh
sh create_Phase_data1.sh
```
This step can take several hours depending on hardware's computational capabilities. 

To train the network on binary configuration run
```sh
sh train_1.sh T1 {contrast_name} Mask {gpu_ID} binary
sh train_2.sh T1 {contrast_name} Mask {gpu_ID} binary
```
where contrast_name is either "QSM" or "Phase", and gpu_ID is the ID of the Cuda visible device to use.

To train the network on multiclass configuration run
```sh
sh train_1.sh T1 QSM Mask {gpu_ID} multiclass
sh train_2.sh T1 QSM Mask {gpu_ID} multiclass
```

To test the network on binary configuration run
sh test_1.sh T1 Phase/QSM Mask 4 binary
sh test_1.sh T1 Phase/QSM Mask 4 binary


## Code Contributors

This work is part of the project MSxplain, and has been submitted to Radiology: Artificial Intelligence.

## Author

👤 **Federico Spagnolo**

- Github: [@federicospagnolo](https://github.com/federicospagnolo)
- | [LinkedIn](https://www.linkedin.com/in/federico-spagnolo/) |

## References

1. 
