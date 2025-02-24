import pytest
import torch
from src.modular_rimnet import ConvBlock

# Assuming your_module.py is where the ConvBlock class is defined


def test_conv_block_single_values():
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    padding = 1
    activation = torch.nn.ReLU()
    num_layers = 3
    use_batchnorm = True
    dropout_rate = 0.2

    block = ConvBlock(
        in_channels,
        out_channels,
        kernel_size,
        padding,
        activation,
        num_layers,
        use_batchnorm,
        dropout_rate,
    )

    assert (
        len(block.conv_block) == num_layers * 4
    )  # Check if correct number of layers are added a layer per Conv, Activation, BN and dropout

    # Add more assertions based on your specific expectations


def test_conv_block_lists():
    in_channels = 3
    out_channels = 64
    kernel_size = [3, 3, 3]
    padding = [1, 1, 1]
    activation = [torch.nn.ReLU(), torch.nn.Tanh(), torch.nn.LeakyReLU()]
    num_layers = 3
    use_batchnorm = [True, False, True]
    dropout_rate = [0.2, 0.0, 0.3]

    block = ConvBlock(
        in_channels,
        out_channels,
        kernel_size,
        padding,
        activation,
        num_layers,
        use_batchnorm,
        dropout_rate,
    )
    assert len(block.conv_block) == num_layers + sum(
        [act is not None for act in activation]
    ) + sum([bn for bn in use_batchnorm]) + sum(
        [dp > 0 for dp in dropout_rate]
    )  # Check if correct number of layers are added


def test_conv_block_forward_pass():
    in_channels = 3
    out_channels = 64
    kernel_size = [3, 3, 3]
    padding = [1, 1, 1]
    activation = [torch.nn.ReLU(), torch.nn.Tanh(), torch.nn.LeakyReLU()]
    num_layers = 3
    use_batchnorm = [True, False, True]
    dropout_rate = [0.2, 0.0, 0.3]

    block = ConvBlock(
        in_channels,
        out_channels,
        kernel_size,
        padding,
        activation,
        num_layers,
        use_batchnorm,
        dropout_rate,
    )

    fix_size = 128
    batch = 2
    input_tensor = torch.randn(batch, in_channels, fix_size, fix_size, fix_size)

    # Perform a forward pass
    output = block(input_tensor)

    # Add assertions based on your specific expectations for the output
    assert output.shape == (batch, out_channels, fix_size, fix_size, fix_size)


import torch


def test_conv_block_forward_pass_gpu():
    # Specify GPU device (you may need to adjust the device index based on your system)
    device = torch.device("cuda:0")

    in_channels = 3
    out_channels = 64
    kernel_size = [3, 3, 3]
    padding = [1, 1, 1]
    activation = [torch.nn.ReLU(), torch.nn.Tanh(), torch.nn.LeakyReLU()]
    num_layers = 3
    use_batchnorm = [True, False, True]
    dropout_rate = [0.2, 0.0, 0.3]

    # Create a ConvBlock instance on GPU
    block = ConvBlock(
        in_channels,
        out_channels,
        kernel_size,
        padding,
        activation,
        num_layers,
        use_batchnorm,
        dropout_rate,
    )
    block.to(device)

    # Assuming your input tensor has the appropriate shape
    fix_size = 128
    batch = 2
    input_tensor = torch.randn(batch, in_channels, fix_size, fix_size, fix_size).to(
        device
    )

    # Perform a forward pass on GPU
    output = block(input_tensor)

    # Add assertions based on your specific expectations for the output
    assert output.shape == (batch, out_channels, fix_size, fix_size, fix_size)
    torch.cuda.empty_cache()
    # Add more assertions as needed
