"""
Created on Tue Jun 18 14:06:57 2024

@author: federico.spagnolo
"""
############
#Usage: python train_rimnet.py FLAIR QSM
############
import sys, os
import argparse
import numpy as np
import nibabel as nib
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from monai.transforms import MapTransform, Compose, Lambda, CenterSpatialCropd, RandRotate90d, RandRotated, RandFlipd, NormalizeIntensityd, RandShiftIntensityd, RandStdShiftIntensityd, RandGaussianSmoothd, RandGaussianNoised, RandGaussianSharpend, Rand3DElasticd, ScaleIntensity, ScaleIntensityd, RandAxisFlipd, RandAffine, RandAffined, RandSpatialCropd, ThresholdIntensity, RandGaussianNoise, RandStdShiftIntensity, RandAxisFlip, RandRotate90

from src.rimnet_archs_mod import BinaryDeepPRL, MulticlassDeepPRL, MonoDeepPRL
from src.rimnet_archs import BimodalRimNet, MonomodalRimNet
from src.trainer import BasicRimNetTrainer
from src.SMSC import PatchesFromCSVCached
#from src.imbalanced import ImbalancedDatasetSampler

class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        """
        Focal Loss to address class imbalance.
        
        Parameters:
        - gamma: focusing parameter to reduce the relative loss for well-classified examples (default=2.0).
        - alpha: weights //to balance classes (default=None). When < 0.5 gives more importance to the negative class.
        - reduction: specifies the reduction to apply to the output ('none', 'mean', or 'sum').
        """
        super(FocalLoss, self).__init__()
        self.gamma = torch.tensor(gamma, dtype = torch.float32).cuda()
        self.alpha = torch.tensor(alpha, dtype = torch.float32).cuda()

    def forward(self, inputs, targets):
        
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class ClassBasedTransforms(MapTransform):
    def __init__(self, prob):
        self.prob = prob
        self.transformPRL = RandSpatialCropd(keys=["img"], roi_size=input_size, random_size=False, random_center=True)
        #self.transformWML = transform-toadd

    def __call__(self, data):
        if data["label"] == 1:
             data = self.transformPRL(data)
        #else:
        #     data = self.transformWML(data)     
        return data
             
class ContrastBasedTransforms(MapTransform):
    def __init__(self, prob):
        self.transform = Compose([
                                 #ScaleIntensity(minv=0.0, maxv=1.0, channel_wise=True),
                                 RandGaussianNoise(std=.01, prob=prob),
                                 RandStdShiftIntensity(factors=.1, channel_wise=False, prob=prob),
                                 RandAffine(translate_range=1, rotate_range=(-3*np.pi/180, 3*np.pi/180), prob=prob)
                                ])
    def __call__(self, data):
        #data["img"][0] = self.transform(data["img"][0]) # t1
        data["img"][1] = self.transform(data["img"][1]) # qsm/phase 1/2 if mask is input**************************
        
        return data


if __name__ == "__main__":
    import warnings 
    # Settings the warnings to be ignored 
    warnings.filterwarnings('ignore')
    current_dir = os.getcwd()
    input_size = (28, 28, 28)
    
    ###################################
    folder = sys.argv[1] # choose between fold_1....fold_5
    base_contrast = sys.argv[2] # choose between FLAIR and T1
    ph_contrast = sys.argv[3] # choose between Phase and QSM
    task = sys.argv[-1]
    if len(sys.argv) > 6:
        if task == "deepPRLmono":
            mask_contrast = sys.argv[4]
            modalities = [f"{ph_contrast}_{mask_contrast}", f"cont_{base_contrast}"]
            sub_folder = sys.argv[5] # choose between sub_fold_1....sub_fold_3
        else:    
            third_contrast = sys.argv[4] # choose a third contrast (Mask)
            mask_contrast = f"{third_contrast}"
            modalities = [f"{ph_contrast}_{third_contrast}", f"cont_{base_contrast}", f"cont_{ph_contrast}"]
            sub_folder = sys.argv[5]
    elif len(sys.argv) > 5:
        mask_contrast = ""
        modalities = [f"cont_{base_contrast}", f"cont_{ph_contrast}"]
        sub_folder = sys.argv[4] # choose between sub_fold_1....sub_fold_3
    elif len(sys.argv) > 4:
        folder = sys.argv[1] # choose between fold_1....fold_5
        base_contrast = sys.argv[2] # choose between FLAIR and T1
        ph_contrast = "QSM"
        mask_contrast = ""
        modalities = [f"cont_{base_contrast}"]
        sub_folder = sys.argv[3] # choose between sub_fold_1....sub_fold_3
    else:
        print("Inputs number incorrect")
        sys.exit()
    ###################################
 
    # Manually setting arguments
    class Args:
        if task == "rimnetmono":
             experiment_out = f'{current_dir}/training/{task}/{folder}/{sub_folder}/{base_contrast}'   
        elif mask_contrast == "":
             experiment_out = f'{current_dir}/training/{task}/{folder}/{sub_folder}/{ph_contrast}_{base_contrast}'
        else:
             experiment_out = f'{current_dir}/training/{task}/{folder}/{sub_folder}/{ph_contrast}_{base_contrast}_{mask_contrast}'
             
        if base_contrast == 'FLAIR' or base_contrast == 'T1':
             csv_path_train = f'{current_dir}/INsIDER_MS_patches_nested_kfold_binary/{ph_contrast}/{folder}/{sub_folder}/train/train_data.csv' #**************
             csv_path_val = f'{current_dir}/INsIDER_MS_patches_nested_kfold_binary/{ph_contrast}/{folder}/{sub_folder}/val/val_data.csv'
        else:      
             csv_path_train = f'{current_dir}/INsIDER_MS_patches_nested_kfold_{task}/{ph_contrast}/{folder}/{sub_folder}/train/train_data.csv'
             csv_path_val = f'{current_dir}/INsIDER_MS_patches_nested_kfold_{task}/{ph_contrast}/{folder}/{sub_folder}/val/val_data.csv'
        # change file location in patches from csv cached
        no_transform = False
        transform_prob = 0.5
        imbalance_sampler = True
        epochs = 80 #80 for QSM 100 for phase
        batch_size = 32
        lr = 0.0001
        weight_decay = 3e-5

    args = Args()

    ''' Fix seed '''
    seed = 12
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    batch = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    print(
            f"Modalities in the CSV to be employed {modalities}"
        )
    sampler = None 
    prob = args.transform_prob
    
    if args.no_transform:
        transform = None
    else:
        transform = Compose([
                                 RandAxisFlipd(keys=["img"], prob=prob),
                                 RandRotate90d(keys=["img"], prob=prob), 
                                 RandAffined(keys=["img"], translate_range=3, rotate_range=(-5*np.pi/180, 5*np.pi/180), prob=prob),
                                 ContrastBasedTransforms(prob=prob),
                                 #ClassBasedTransforms(prob=prob),
                                 CenterSpatialCropd(keys=["img"], roi_size=input_size)
                                 ]).set_random_state(seed=seed)

    dataset_train = PatchesFromCSVCached(
        own_transform = transform,
        csv_path=f'{args.csv_path_train}',
        use_modalities=modalities,
        task=task
    )
    
    dataset_val = PatchesFromCSVCached(
        csv_path=f'{args.csv_path_val}',
        use_modalities=modalities,
        transform=Compose(CenterSpatialCropd(keys=["img"], roi_size=28)).set_random_state(seed=seed),
        task=task
    )

    if args.imbalance_sampler:
        labels_series = dataset_train.get_labels()
        if task == "multiclass":
             labels = labels_series.map(lambda label: 1 if label in {1, 7} else 2 if label in {2, 8} else 0)
             weights = 1./np.array(labels.value_counts().reindex([0, 1, 2]))
        else:
             labels = labels_series.map(lambda label: 1 if label in {1, 7} else 0)
             #labels = labels_series.map(lambda label: 1 if label in {1, 2, 7, 8} else 0)
             weights = 1./np.array(labels.value_counts())
        class_weights = torch.FloatTensor(weights).to(device)
        print(f"Setting sampling weights: {weights}")
        samples_weight = torch.from_numpy(np.array([weights[t] for t in labels])).double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True) 
        
        # Create a subset of indices sampled by the sampler
        sampled_indices = list(sampler)
        sampled_dataset_train = Subset(dataset_train, sampled_indices)
        
        val_labels_series = dataset_val.get_labels()
        if task == "multiclass":
             val_labels = val_labels_series.map(lambda label: 1 if label in {1, 7} else 2 if label in {2, 8} else 0)
             val_weights = 1./np.array(val_labels.value_counts().reindex([0, 1, 2]))
        else:
             val_labels = val_labels_series.map(lambda label: 1 if label in {1, 7} else 0)
             #val_labels = val_labels_series.map(lambda label: 1 if label in {1, 2, 7, 8} else 0)
             val_weights = 1./np.array(val_labels.value_counts())
        val_weights = val_weights / val_weights.sum()
        val_class_weights = torch.FloatTensor(val_weights).to(device)
        # Higher weights to minority class 
        print(f"Setting loss weights: {val_weights}")
    
    train_dataloader = DataLoader(
        sampled_dataset_train,
        #dataset_train,
        batch_size=batch,
        shuffle=False if args.imbalance_sampler else True,
        sampler=sampler,
        num_workers=4,
        #worker_init_fn=worker_init_fn,
    )
    val_dataloader = DataLoader(
        dataset_val,
        batch_size=batch,
        shuffle=False,
        #sampler=sampler,
        num_workers=2,
        #worker_init_fn=worker_init_fn,
    )
    
    #### Store transforms
    #i = 0
    #for batch_idx, batch in enumerate(train_dataloader):
    #    for img, label, subject, lesion in zip(batch["img"], batch["label"], batch["subject"], batch["lesion"]):
    #         output_path = '/home/federico.spagnolo/Federico/PRLsConfluent/train_AB_FS/transforms'
    #         nifti_mask = nib.Nifti1Image(np.array(img[0]), affine=np.eye(4))
    #         nib.save(nifti_mask, f'{output_path}/{subject}_{lesion}_{label}_{i}_mask.nii.gz')
    #         nifti_t1 = nib.Nifti1Image(np.array(img[1]), affine=np.eye(4))
    #         nib.save(nifti_t1, f'{output_path}/{subject}_{lesion}_{label}_{i}_t1.nii.gz')
    #         nifti_qsm = nib.Nifti1Image(np.array(img[2]), affine=np.eye(4))
    #         nib.save(nifti_qsm, f'{output_path}/{subject}_{lesion}_{label}_{i}_qsm.nii.gz')
    #         i = i + 1
                     
    #sys.exit()
    ##########################################
    
    if task == "multiclass":
           model = MulticlassDeepPRL(input_size, batchnorm=True).to(device)
    elif task == "rimnet":
           model = BimodalRimNet(input_size, batchnorm=True).to(device)
    elif task == "rimnetmono":
           model = MonomodalRimNet(input_size, batchnorm=True).to(device)            
    elif task == "binary":
           model = BinaryDeepPRL(input_size, batchnorm=True).to(device)
    elif task == "deepPRLmono":
           model = MonoDeepPRL(input_size, batchnorm=True).to(device)              
    else:
           assert("Too many or too few arguments!")
    
    #print(model)
    
    ##########################################
    
    focal_loss_fn = FocalLoss(gamma=2.0, alpha=0.2)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    trainer = BasicRimNetTrainer(
        model,
        train_dataloader,
        val_dataloader,
        task=task,
        output_path=args.experiment_out,
        num_epochs=args.epochs,
        optimizer=optimizer,
        loss_fn = focal_loss_fn,
        #loss_fn = torch.nn.CrossEntropyLoss(weight=val_class_weights, reduction='mean'),
        lr_scheduler=PolynomialLR(optimizer, args.epochs, power = 0.9),
        device=device
    )
    
    trainer.train()
