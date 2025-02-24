import argparse
import tempfile

import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from monai.transforms import Compose, CenterSpatialCropd, RandRotate90d, RandRotated, RandFlipd, NormalizeIntensityd, RandShiftIntensityd, RandStdShiftIntensityd, RandGaussianSmoothd,RandGaussianNoised, RandGaussianSharpend, Rand3DElasticd, ScaleIntensityd, RandAxisFlipd, RandAffined, RandSpatialCropd

from src.rimnet_archs import BimodalRimNet
from src.trainer import BasicRimNetTrainer
from src.datasets import PatchesFromCSV, PatchesFromCSVCached
from src.imbalanced import ImbalancedDatasetSampler
from src.loss import  MemoryEfficientSoftDiceLoss


if __name__ == "__main__":
    import warnings 
    # Settings the warnings to be ignored 
    warnings.filterwarnings('ignore') 
    parser = argparse.ArgumentParser(description="Process folds with command-line arguments.")
    parser.add_argument("fold", help="The number of the fold to process.")
    parser.add_argument("--experiment_out", default=tempfile.gettempdir(), help="")
    parser.add_argument("--no_transform",  action="store_true", help="Apply transforms.")
    parser.add_argument("--transform_prob", type=float, default=0.5,help="The number ")
    parser.add_argument("--imbalance_sampler",  action="store_true", help="Apply transforms.")
    parser.add_argument("--epochs", type=int, default=1000,help="The number of the fold to process.")
    parser.add_argument("--batch_size", type=int, default=200,help="The number of the fold to process.")
    parser.add_argument("--lr", type=float, default=3e-4,help="The number of the fold to process.")
    parser.add_argument("--weight_decay", type=float, default=3e-5,help="The number of the fold to process.")
    
    args = parser.parse_args()
    device = torch.device("cuda:0")
    batch = args.batch_size
    lr = args.lr
    weight_decay=args.weight_decay
    modalities = ["Flair", "Phase"]
    sampler = None 
    prob = args.transform_prob
    fix_transform = Compose([CenterSpatialCropd(keys=["img"], roi_size=28),  NormalizeIntensityd(channel_wise=True, keys=["img"])])
    if args.no_transform:
        transform = None
    else:
        transform = Compose([#RandShiftIntensityd(keys=["img"], offsets=2),
                            #Rand3DElasticd(keys=["img"], sigma_range=[1,3], magnitude_range=[-1,1],prob=prob)
                            RandStdShiftIntensityd(keys=["img"], factors=3, channel_wise=True, prob=prob),
                            RandFlipd(keys=["img"], prob=prob), 
                            RandRotate90d(keys=["img"], prob=prob), 
                            RandRotated(keys=["img"], range_x=[0.4,0.4], prob=prob),
                            #RandGaussianNoised(keys=["img"], prob=prob), 
                            #RandGaussianSmoothd(keys=["img"], prob=prob), 
                            #RandGaussianSharpend(keys=["img"], prob=prob), 
        ] )
        joe_transform = Compose([#Rand3DElasticd(keys=["img"], sigma_range=[1,5], magnitude_range=[1,15],prob=prob),
                                 RandGaussianNoised(keys=["img"], prob=prob),
                                 RandStdShiftIntensityd(keys=["img"], factors=3, channel_wise=True, prob=prob),
                                 ScaleIntensityd(keys=["img"],minv=0.0, maxv=1.0, channel_wise=True),
                                 RandAxisFlipd(keys=["img"], prob=prob), 
                                 RandRotate90d(keys=["img"], prob=prob), 
                                 RandAffined(keys=["img"],translate_range=3,prob=prob),
                                 RandSpatialCropd(keys=["img"],roi_size=(28,28,28))]) 

    dataset_train = PatchesFromCSVCached(
        own_transform = joe_transform,
        csv_path=f'/code/TorchRimNet/data/new_patches_basel_chuv/fold_{args.fold}_train.csv',
        use_modalities=modalities,
        transform=Compose([ScaleIntensityd(keys=["img"],minv=0.0, maxv=1.0, channel_wise=True)]),
    )
 
    dataset_test = PatchesFromCSVCached(
        csv_path=f'/code/TorchRimNet/data/new_patches_basel_chuv/fold_{args.fold}_test.csv',
        use_modalities=modalities,
        transform=Compose([ScaleIntensityd(keys=["img"],minv=0.0, maxv=1.0, channel_wise=True),
                           CenterSpatialCropd(keys=["img"], roi_size=28)]),
    )

    if args.imbalance_sampler:
        labels = dataset_train.get_labels()
        weight = 1./np.array(labels.value_counts())
        samples_weight = torch.from_numpy(np.array([weight[t] for t in labels])).double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight)) 
    
    train_dataloader = DataLoader(
        dataset_train,
        batch_size=batch,
        shuffle=False if args.imbalance_sampler else True,
        sampler=sampler
        #num_workers=1,
        #worker_init_fn=worker_init_fn,
    )
    val_dataloader = DataLoader(
        dataset_test,
        batch_size=35,
        shuffle=False,
        #num_workers=1,
        #worker_init_fn=worker_init_fn,
    )
    input_size = (28, 28, 28)
    model = BimodalRimNet(input_size, batchnorm=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    trainer = BasicRimNetTrainer(
        model,
        train_dataloader,
        val_dataloader,
        output_path=args.experiment_out,
        num_epochs=args.epochs,
        optimizer=optimizer,
        #loss_fn= MemoryEfficientSoftDiceLoss(torch.nn.Softmax(dim=1), smooth=1e-5),
        #lr_scheduler=PolynomialLR(optimizer, args.epochs, power = 0.9),
        device=device,
    )
    trainer.train()