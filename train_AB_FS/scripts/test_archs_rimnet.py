import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from torchinfo import summary


# tests/test_bimodal_rimnet.py
from src.rimnet_archs import (
    BimodalRimNet,
    BimodalRimNetPlus,
    MonomodalRimNet,
    TrimodalRimNet,
    TrimodalSerendipityRimNet,
)

# from torchinfo import summary


def synthetic_data(modalities, size, samples=50, classes=3, standarize=True):
    # Generate synthetic data (replace this with your actual data generation logic)
    input_size = (size, size, size)  # Adjust according to your input size
    data = torch.randn(samples, modalities, *input_size)  # Move data to GPU
    targets = torch.randint(
        0, classes, (samples,), dtype=torch.long
    )  # Move targets to GPU

    return data, targets


def test_bimodalrimnet():
    input_size = (28, 28, 28)  # Provide the appropriate input size
    model = BimodalRimNet(input_size)

    # Example assertions (update with your actual expectations)
    assert model.n_modalities == 2
    assert len(model.conv_blocks) == 3
    assert model.n_extracted_features == 128


def test_monomodalrimnet():
    input_size = (28, 28, 28)  # Provide the appropriate input size
    model = MonomodalRimNet(input_size)

    # Example assertions (update with your actual expectations)
    assert model.n_modalities == 1
    assert len(model.conv_blocks) == 3
    assert model.n_extracted_features == 64


@pytest.fixture
def data_bacth_2modalities():
    torch.manual_seed(69)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    n_modalities = 2
    n_classes = 2
    batch = 32
    data, targets = synthetic_data(n_modalities, 28, classes=n_classes, samples=batch)
    return data, targets, device


def test_bimodalrimnet_train(data_bacth_2modalities):
    input_size = (28, 28, 28)
    data, targets, device = data_bacth_2modalities
    model = BimodalRimNet(input_size).to(device)
    summary(model)
    print("Targets:", targets)
    data = data.to(device)
    targets = targets.to(device)
    # Define loss function and optimizer with L2 regularization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=0.01, weight_decay=1e-4
    )  # Adjust weight_decay as needed

    # Training loop
    num_epochs = 100  # Adjust as needed
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(data)
        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}")
    assert loss.item() < 0.001
    torch.cuda.empty_cache()


def test_bimodalrimnet_train():
    torch.manual_seed(69)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    input_size = (28, 28, 28)
    model = BimodalRimNet(input_size, batchnorm=True).to(device)
    summary(model)
    n_modalities = 2
    n_classes = 2
    batch = 32
    data, targets = synthetic_data(n_modalities, 28, classes=n_classes, samples=batch)
    print("Targets:", targets)
    data = data.to(device)
    data = torch.nn.functional.normalize(data, dim=1)
    targets = targets.to(device)
    # Define loss function and optimizer with L2 regularization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=0.01, weight_decay=1e-4
    )  # Adjust weight_decay as needed

    # Training loop
    num_epochs = 100  # Adjust as needed
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(data)
        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}")
    assert loss.item() < 0.001
    torch.cuda.empty_cache()


def test_trimodalserendipityrimnet_train():
    torch.manual_seed(69)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    input_size = (28, 28, 28)
    model = TrimodalSerendipityRimNet(input_size, batchnorm=True).to(device)
    summary(model)
    n_modalities = 3
    n_classes = 2
    batch = 32
    data, targets = synthetic_data(n_modalities, 28, classes=n_classes, samples=batch)
    print("Targets:", targets)
    data = data.to(device)
    data = torch.nn.functional.normalize(data, dim=1)
    targets = targets.to(device)
    # Define loss function and optimizer with L2 regularization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=0.01, weight_decay=1e-4
    )  # Adjust weight_decay as needed

    # Training loop
    num_epochs = 100  # Adjust as needed
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(data)
        # Compute the loss
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}")
    assert loss.item() < 0.001
    torch.cuda.empty_cache()
