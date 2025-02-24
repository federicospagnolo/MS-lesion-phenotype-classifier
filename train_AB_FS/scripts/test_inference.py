import os
import pandas as pd
import pytest
import torch
from monai.transforms import Compose, CenterSpatialCropd

from src.rimnet_archs import BimodalRimNet
from src.datasets import PatchesFromCSV
from src.inference import BasicRimNetInferer

current_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def csv_test():
    csv_path = os.path.join(current_dir, "test_data", "random_csv.csv")
    return csv_path


@pytest.fixture
def model_bimodal_test():
    input_size = (28, 28, 28)
    return BimodalRimNet(input_size, batchnorm=True)


def test_BimodalRimNet_inference(csv_test, model_bimodal_test):
    checkpoint_path = os.path.join(
        current_dir,
        "test_data",
        "exp_BimodalRimNet_5_2_2024_23_24_16",
        "checkpoint_best.pth",
    )

    device = torch.device("cuda:0")

    batch = 5
    modalities = ["modality1", "modality2"]
    dataset = PatchesFromCSV(
        csv_path=csv_test,
        use_modalities=modalities,
        transform=Compose(CenterSpatialCropd(keys=["img"], roi_size=28)),
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch, shuffle=False, num_workers=6
    )
    inferer = BasicRimNetInferer(checkpoint_path, model_bimodal_test, device=device)
    result_df = inferer(test_dataloader)

    # Add your assertions based on the expected behavior of the inference
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == len(test_dataloader.dataset)
