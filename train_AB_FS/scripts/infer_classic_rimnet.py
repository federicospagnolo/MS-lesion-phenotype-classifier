import os
import argparse
import torch
from monai.transforms import Compose, CenterSpatialCropd, ScaleIntensityd
from src.rimnet_archs import BimodalRimNet
from src.inference import BasicRimNetInferer
from src.SMSC import PatchesFromCSV

if __name__ == "__main__":
    # Ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process folds with command-line arguments.")
    parser.add_argument("model_path", help="Path to the model weights.")
    parser.add_argument("csv_path", help="Path to the CSV file containing patch paths.")
    parser.add_argument("save_results_path", help="Path to save the results in a CSV.")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for inference.")
    parser.add_argument("--device", default="cuda:0", help="Device for inference.") 
    parser.add_argument("--modalities", nargs='+', default=["cont_FLAIR", "cont_T2STAR_PHASE"], help="List of modalities to use.")
    parser.add_argument("--num_workers", default=6, type=int, help="Number of workers for the data loader.")
    args = parser.parse_args()

    # Create dataset
    transform = Compose([
        ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0, channel_wise=True),
        CenterSpatialCropd(keys=["img"], roi_size=28)
    ])
    dataset = PatchesFromCSV(csv_path=args.csv_path, use_modalities=args.modalities, transform=transform)

    # Create DataLoader for inference
    inference_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Create inference engine
    model = BimodalRimNet((28, 28, 28), batchnorm=True)
    inferer = BasicRimNetInferer(args.model_path, model, device=args.device)

    # Perform inference and save results
    results = inferer(inference_dataloader)
    results.to_csv(args.save_results_path)
