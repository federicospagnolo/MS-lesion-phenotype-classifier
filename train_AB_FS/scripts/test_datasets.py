import os
import pytest
import torch
from torch.utils.data import DataLoader
from src.datasets import (
    PatchesFromCSV,
)
from monai.transforms import Compose, FlipD


@pytest.fixture
def csv_test():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "test_data", "random_csv.csv")
    return csv_path


def test_patches_from_csv(csv_test):
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # csv_path = os.path.join(current_dir, "test_data", "random_csv.csv")
    modalities = ["modality1", "modality2", "modality3"]
    transform = None  # You can add transformations here if needed

    dataset = PatchesFromCSV(
        csv_path=csv_test, use_modalities=modalities, transform=transform
    )

    # Check length of the dataset
    assert len(dataset) > 0, "Dataset is empty"

    # Check each sample in the dataset
    for i in range(len(dataset)):
        sample = dataset[i]

        # Check if the required keys exist in the sample
        assert "img" in sample, "Key 'img' not found in the sample"
        if not dataset.deployment_dataset:
            assert "label" in sample, "Key 'label' not found in the sample"

        # Check the shapes of the image tensor
        img_tensor = sample["img"]
        assert isinstance(img_tensor, torch.Tensor), "Image should be a PyTorch Tensor"
        assert (
            img_tensor.dim() == 4
        ), "Image tensor should be 4-dimensional (C, D, H, W)"
        assert img_tensor.shape == (3, 34, 34, 34)  # 3 modalities

        # Additional checks can be added based on your specific requirements


def test_patches_from_csv_modalities_order(csv_test):
    modalities = ["modality2", "modality1"]
    transform = None  # You can add transformations here if needed

    dataset = PatchesFromCSV(
        csv_path=csv_test, use_modalities=modalities, transform=transform
    )
    for modality in range(len(modalities)):
        assert modalities[modality] == dataset.modalities[modality]


def test_patches_from_csv_extra_info(csv_test):
    modalities = ["modality1", "modality2", "modality3"]
    extra_info = ["Extra_Column1"]
    transform = None  # You can add transformations here if needed
    dataset = PatchesFromCSV(
        csv_path=csv_test,
        use_modalities=modalities,
        transform=transform,
        keep_extra_cols=extra_info,
    )
    print("DF size", len(dataset.df), len(dataset))
    for drawn in range(len(dataset)):
        print(dataset[drawn]["Extra_Column1"])


def test_patches_from_csv_with_monai_transforms(csv_test):
    modalities = ["modality1", "modality2", "modality3"]
    transform = Compose([FlipD(keys=["img"], spatial_axis=0)])
    dataset = PatchesFromCSV(
        csv_path=csv_test, use_modalities=modalities, transform=transform
    )

    # Test __getitem__ method
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert (
        "img" in sample
    )  # Assuming "img" is the key for images in the drawn dictionary
    assert FlipD(keys=["img"], spatial_axis=0)
