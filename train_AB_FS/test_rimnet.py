# python test_rimnet.py --modalities T1 QSM Mask --folder 1 --sub_folder 1
import os, sys
import argparse
import torch
from glob import glob
from monai.transforms import Compose, CenterSpatialCropd, ScaleIntensityd
from src.rimnet_archs import BimodalRimNet
from src.rimnet_archs_mod import TrimodalRimNet, MulticlassDeepPRL
from src.XAI import BasicRimNetInferer, BasicRimNetXAI
from src.SMSC import PatchesFromCSVCached

current_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # Ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    input_size = (28, 28, 28)

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process folds with command-line arguments.")
    #parser.add_argument("--best_epoch", type=int, help="Best epoch number.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--device", default="cuda", help="Device for inference.")
    parser.add_argument("--task", default="multiclass", help="Choose between binary and multiclass.")
    parser.add_argument("--modalities", nargs='+', default=["T1", "QSM", "Mask"], help="Input MRI contrasts")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of workers for the data loader.")
    parser.add_argument("--folder", type=int, help="Test folder, integer from 1 to 5")
    parser.add_argument("--sub_folder", type=int, help="Model sub-folder, integer from 1 to 3")
    args = parser.parse_args()
    
    sub_folder = f"sub_fold_{args.sub_folder}"
    base_contrast = args.modalities[0]
    ph_contrast = args.modalities[1]
    folder = f"fold_{args.folder}"
    if len(args.modalities) > 2:
         third_contrast = args.modalities[2]
         model_folder = glob(f'/home/federico.spagnolo/Federico/PRLsConfluent/train_AB_FS/training/{args.task}/{folder}/{sub_folder}/{ph_contrast}_{base_contrast}_{third_contrast}/*')[0]
         modalities = [f"{ph_contrast}_{third_contrast}", f"cont_{base_contrast}", f"cont_{ph_contrast}"]
    elif len(args.modalities) == 2:
         model_folder = glob(f'/home/federico.spagnolo/Federico/PRLsConfluent/train_AB_FS/training/{args.task}/{folder}/{sub_folder}/{ph_contrast}_{base_contrast}/*')[0]
         modalities = [f"cont_{base_contrast}", f"cont_{ph_contrast}"]
    else:
         print("Wrong n. of modalities")
         sys.exit()     
    model_path = os.path.join(model_folder, f"checkpoint_best_Epoch.pth")
    test_folder = f'/home/federico.spagnolo/Federico/PRLsConfluent/train_AB_FS/test/{args.task}/{ph_contrast}_{base_contrast}_nested_kfold/{folder}'
    csv_path = f'/home/federico.spagnolo/Federico/PRLsConfluent/train_AB_FS/INsIDER_MS_patches_nested_kfold_{args.task}/{folder}/test/test_data.csv'
    #results_path = '/home/federico.spagnolo/Federico/PRLsConfluent/train_AB_FS/xai_results'
    
    # Create dataset
    dataset = PatchesFromCSVCached(
        csv_path=csv_path,
        use_modalities=modalities,
        transform=CenterSpatialCropd(keys=["img"], roi_size=28),
        task=args.task
    )
    

    # Create DataLoader for inference
    test_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    print(f"Using device {args.device}")
    
    # Create inference engine
    if args.task == "multiclass":
           model = MulticlassDeepPRL(input_size, batchnorm=True).to(args.device)
    elif len(modalities) == 2:
           model = BimodalRimNet(input_size, batchnorm=True).to(args.device)
    elif len(modalities) == 3:
           model = TrimodalRimNet(input_size, batchnorm=True).to(args.device)       
           
    inferer = BasicRimNetInferer(model_path, model, task=args.task, device=args.device)
    #inferer = BasicRimNetXAI(os.path.dirname(csv_path), results_path, model_path, model, device=args.device)

    # Perform inference and save results
    results = inferer(test_dataloader, test_folder)
    #results = inferer(test_dataloader)
