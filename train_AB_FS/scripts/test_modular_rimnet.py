import pytest
import torch
import torch.nn as nn
import torch.optim as optim

# from torchsummary import summary  # Assuming you have torchsummary installed
from torchinfo import summary
from src.modular_rimnet import ModularRimNet, ConvBlock


def test_modular_rimnet():
    input_size = (28, 28, 28)
    n_modalities = 3
    n_classes = 2

    conv_blocks = [
        {
            "in_channels": 1,
            "out_channels": 32,
            "kernel_size": 3,
            "padding": 1,
            "activation": torch.nn.ReLU(),
            "num_layers": 3,
            "use_batchnorm": True,
            "dropout_rate": 0.2,
        },
        {
            "in_channels": 32,
            "out_channels": 64,
            "kernel_size": 3,
            "padding": 1,
            "activation": torch.nn.ReLU(),
            "num_layers": 3,
            "use_batchnorm": True,
            "dropout_rate": 0.2,
        },
        {
            "in_channels": 64,
            "out_channels": 128,
            "kernel_size": 3,
            "padding": 1,
            "activation": torch.nn.ReLU(),
            "num_layers": 3,
            "use_batchnorm": True,
            "dropout_rate": [0.1, 0.2],
        },
    ]

    poolings = [
        torch.nn.MaxPool3d(kernel_size=2, ceil_mode=True)
        for _ in range(len(conv_blocks))
    ]
    reductional_layer = torch.nn.Conv3d(128, 1, 1)
    fully_connected = [512, 256, 128, 64]

    model = ModularRimNet(
        input_size=input_size,
        n_modalities=n_modalities,
        conv_blocks=conv_blocks,
        poolings=poolings,
        reduction_layer=reductional_layer,
        fully_connected=fully_connected,
        n_classes=n_classes,
    )

    # for modality in range(model.n_modalities):
    #    print(f"-----Modality: {modality}----------")
    #    print(model.subnet_for_modality[modality])
    # for i, fc in enumerate(model.fcl):
    #    print(f"FC {i}:", fc)
    summary(model)
    print("Output", model.convolutional_features_shape, model.n_extracted_features)

    # Assuming your input tensor has the appropriate shape
    # fix_size = 64
    # batch = 2
    # input_tensor = torch.randn(n_modalities, fix_size, fix_size, fix_size)

    # Print model summary using torchsummary
    # summary(model, input_tensor.shape)

    # Perform assertions or additional tests based on your specific expectations


def test_modular_rimnet_forward():
    input_size = (32, 32, 32)
    n_modalities = 2
    conv_blocks = [
        {
            "in_channels": 1,
            "out_channels": 64,
            "kernel_size": 3,
            "padding": 1,
            "activation": torch.nn.ReLU(),
            "num_layers": 2,
            "use_batchnorm": True,
            "dropout_rate": 0.2,
        },
        {
            "in_channels": 64,
            "out_channels": 128,
            "kernel_size": 3,
            "padding": 1,
            "activation": torch.nn.ReLU(),
            "num_layers": 2,
            "use_batchnorm": True,
            "dropout_rate": 0.2,
        },
    ]
    poolings = [torch.nn.MaxPool3d(kernel_size=2) for _ in range(len(conv_blocks))]
    reduction_layer = torch.nn.Conv3d(128, 1, 1)
    fully_connected = [256, 128]
    n_classes = 10

    model = ModularRimNet(
        input_size,
        n_modalities,
        conv_blocks,
        poolings,
        reduction_layer,
        fully_connected,
        n_classes,
    )
    summary(model)

    fix_size = 32
    batch = 2
    # summary(model, input_size=(batch,n_modalities, fix_size, fix_size, fix_size))
    # Create a random input tensor with the specified batch size and number of modalities
    input_tensor = torch.randn(batch, n_modalities, fix_size, fix_size, fix_size)

    # Perform a forward pass
    output = model(input_tensor)

    # Add assertions based on your specific expectations for the output
    assert output.shape == (batch, n_classes)


def test_modular_rimnet_initialization():
    # Define the model configuration
    input_size = (28, 28, 28)
    n_modalities = 3
    n_classes = 2

    conv_blocks = [
        {
            "in_channels": 1,
            "out_channels": 32,
            "kernel_size": 3,
            "padding": 1,
            "activation": torch.nn.ReLU(),
            "num_layers": 3,
            "use_batchnorm": True,
            "dropout_rate": 0,
        },
        {
            "in_channels": 32,
            "out_channels": 64,
            "kernel_size": 3,
            "padding": 1,
            "activation": torch.nn.ReLU(),
            "num_layers": 3,
            "use_batchnorm": True,
            "dropout_rate": 0,
        },
        {
            "in_channels": 64,
            "out_channels": 128,
            "kernel_size": 3,
            "padding": 1,
            "activation": torch.nn.ReLU(),
            "num_layers": 3,
            "use_batchnorm": False,
            "dropout_rate": 0,
        },
    ]

    poolings = [
        torch.nn.MaxPool3d(kernel_size=2, ceil_mode=True)
        for _ in range(len(conv_blocks))
    ]
    reduction_layer = torch.nn.Conv3d(128, 1, 1)
    fully_connected = [512, 256, 128, 64]

    # Create an instance of your ModularRimNet model
    model = ModularRimNet(
        input_size=input_size,
        n_modalities=n_modalities,
        conv_blocks=conv_blocks,
        poolings=poolings,
        reduction_layer=reduction_layer,
        fully_connected=fully_connected,
        n_classes=n_classes,
    )

    # Initialize the layers
    # model.initialize_layers()

    # Check the initialization of convolutional layers
    # this is calculated from xavier uniform initialization with a (3,3,3) kernel size
    a = 0.5773502691896257  # sqrt(1/3)
    for s, subnet in enumerate(model.subnet_for_modality):
        for b, block in enumerate(subnet):
            if isinstance(block, nn.Sequential):
                for sb, seq_block in enumerate(block):
                    if isinstance(seq_block, ConvBlock):
                        for cb, layer in enumerate(seq_block.conv_block):
                            if isinstance(layer, nn.Conv3d):
                                for param_name, param in layer.named_parameters():
                                    if "bias" in param_name:
                                        assert torch.allclose(
                                            param, torch.zeros_like(param)
                                        )
                                    if "weight" in param_name:
                                        assert torch.all(param >= -a)
                                        assert torch.all(param <= a)

    # Check the initialization of fully connected layers
    for fc in model.fcl:
        for param_name, param in fc.named_parameters():
            if "bias" in param_name:
                assert torch.allclose(param, torch.zeros_like(param))


def synthetic_data(modalities, size, samples=50, classes=3):
    # Generate synthetic data (replace this with your actual data generation logic)
    input_size = (size, size, size)  # Adjust according to your input size
    data = torch.randn(samples, modalities, *input_size)  # Move data to GPU
    targets = torch.randint(
        0, classes, (samples,), dtype=torch.long
    )  # Move targets to GPU

    return data, targets


def test_modular_rimnet_training():
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    fix_size = 28
    input_size = (fix_size, fix_size, fix_size)
    n_modalities = 3
    n_classes = 2
    batch = 50
    data, targets = synthetic_data(
        n_modalities, fix_size, classes=n_classes, samples=batch
    )
    data = data.to(device)
    targets = targets.to(device)
    print("data", data.shape, data.device)

    # Your provided model definition

    conv_blocks = [
        {
            "in_channels": 1,
            "out_channels": 32,
            "kernel_size": 3,
            "padding": 1,
            "activation": torch.nn.ReLU(),
            "num_layers": 3,
            "use_batchnorm": True,
            "dropout_rate": 0,
        },
        {
            "in_channels": 32,
            "out_channels": 64,
            "kernel_size": 3,
            "padding": 1,
            "activation": torch.nn.ReLU(),
            "num_layers": 3,
            "use_batchnorm": True,
            "dropout_rate": 0,
        },
        {
            "in_channels": 64,
            "out_channels": 128,
            "kernel_size": 3,
            "padding": 1,
            "activation": torch.nn.ReLU(),
            "num_layers": 3,
            "use_batchnorm": True,
            "dropout_rate": 0,
        },
    ]

    poolings = [
        torch.nn.MaxPool3d(kernel_size=2, ceil_mode=True)
        for _ in range(len(conv_blocks))
    ]
    reduction_layer = torch.nn.Conv3d(128, 1, 1)
    fully_connected = [256, 128, 64]

    # Create an instance of your ModularRimNet model and move it to GPU
    model = ModularRimNet(
        input_size=input_size,
        n_modalities=n_modalities,
        conv_blocks=conv_blocks,
        poolings=poolings,
        reduction_layer=reduction_layer,
        fully_connected=fully_connected,
        n_classes=n_classes,
    )
    model = model.to(device)
    summary(model)
    # print("MODEL", model)

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

    # Your assertion based on overfitting (e.g., low loss on synthetic data)
    print(targets)
    assert loss.item() < 0.01  # Adjust the threshold as needed
