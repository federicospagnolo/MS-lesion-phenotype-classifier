import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import numpy as np

seed = 8
np.random.seed(seed)
import random

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# For tensorflow
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)

# torch.use_deterministic_algorithms(True)
import pytest

import torch.nn as nn
import torch.optim as optim

# from torchsummary import summary  # Assuming you have torchsummary installed
from torchinfo import summary
from src.modular_rimnet import ModularRimNet
from src.trainer import BasicRimNetTrainer
from src.datasets import PatchesFromCSV
from src.rimnet_archs import BimodalRimNet, TrimodalRimNet, TrimodalSerendipityRimNet
from monai.transforms import Compose, CenterSpatialCropd
import os


@pytest.fixture
def csv_test():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "test_data", "random_csv.csv")
    return csv_path


@pytest.fixture
def model_test():
    input_size = (28, 28, 28)
    return BimodalRimNet(input_size, batchnorm=True)


@pytest.fixture
def model_test_noxavier():
    input_size = (28, 28, 28)
    return BimodalRimNet(input_size, batchnorm=True, xavier_init=False)


@pytest.fixture
def trimodel_test():
    input_size = (28, 28, 28)
    return TrimodalRimNet(input_size, batchnorm=True)


def test_trainer_init(csv_test, model_test):
    device = torch.device("cuda:0")

    batch = 3

    modalities = ["modality1", "modality2"]
    transform = None  # You can add transformations here if needed

    dataset = PatchesFromCSV(
        csv_path=csv_test, use_modalities=modalities, transform=transform
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch, shuffle=True, num_workers=6
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch, shuffle=True, num_workers=6
    )

    # Your provided model definition
    trainer = BasicRimNetTrainer(
        model_test, train_dataloader, val_dataloader, "/tmp/trainig.txt", device=device
    )


def test_trainer_run(csv_test, model_test):
    def worker_init_fn(worker_id):
        np.random.seed(seed)
        random.seed(seed)

    device = torch.device("cuda:0")

    batch = 10

    modalities = ["modality1", "modality2"]
    transform = None

    dataset = PatchesFromCSV(
        csv_path=csv_test,
        use_modalities=modalities,
        transform=Compose(CenterSpatialCropd(keys=["img"], roi_size=28)),
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=1,
        worker_init_fn=worker_init_fn,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=1,
        worker_init_fn=worker_init_fn,
    )

    trainer = BasicRimNetTrainer(
        model_test,
        train_dataloader,
        val_dataloader,
        output_path="/tmp/",
        num_epochs=100,
        optimizer=optim.Adam(model_test.parameters(), lr=0.01, weight_decay=1e-4),
        device=device,
    )
    trainer.train()
    assert trainer.mean_epoch_loss < 0.0001


def test_trainer_trimodal_run(csv_test, trimodel_test):
    def worker_init_fn(worker_id):
        np.random.seed(seed)
        random.seed(seed)

    device = torch.device("cuda:0")

    batch = 10

    modalities = ["modality1", "modality2", "modality3"]
    transform = None

    dataset = PatchesFromCSV(
        csv_path=csv_test,
        use_modalities=modalities,
        transform=Compose(CenterSpatialCropd(keys=["img"], roi_size=28)),
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=1,
        worker_init_fn=worker_init_fn,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=1,
        worker_init_fn=worker_init_fn,
    )

    trainer = BasicRimNetTrainer(
        trimodel_test,
        train_dataloader,
        val_dataloader,
        output_path="/tmp/",
        num_epochs=100,
        optimizer=optim.Adam(trimodel_test.parameters(), lr=0.01, weight_decay=1e-4),
        device=device,
    )
    trainer.train()
    assert trainer.mean_epoch_loss < 0.0012


def test_trainer_run_noxavier(csv_test, model_test_noxavier):
    def worker_init_fn(worker_id):
        np.random.seed(seed)
        random.seed(seed)

    device = torch.device("cuda:0")

    batch = 10

    modalities = ["modality1", "modality2"]
    transform = None

    dataset = PatchesFromCSV(
        csv_path=csv_test,
        use_modalities=modalities,
        transform=Compose(CenterSpatialCropd(keys=["img"], roi_size=28)),
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=1,
        worker_init_fn=worker_init_fn,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=1,
        worker_init_fn=worker_init_fn,
    )

    trainer = BasicRimNetTrainer(
        model_test_noxavier,
        train_dataloader,
        val_dataloader,
        output_path="/tmp/",
        num_epochs=500,
        optimizer=optim.Adam(
            model_test_noxavier.parameters(), lr=0.01, weight_decay=1e-4
        ),
        device=device,
    )
    trainer.train()
    assert trainer.mean_epoch_loss < 0.002
