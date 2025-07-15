# Here are prefixed RimNet configurations. The initialization is fixed to the classic RimNet,
# if you want to change it, just extend whatever RimNet to have the proper :func:`~src.modular_rimnet.initialize_layers`.

import torch.nn as nn
from src.modular_rimnet import ModularRimNet


class BimodalRimNet(ModularRimNet):
    def __init__(self, input_size, batchnorm=False, **kwargs):
        conv_blocks = [
            {
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.Tanh(),
                "num_layers": 2,
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
            {
                "in_channels": 32,
                "out_channels": 64,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.Tanh(),
                "num_layers": 2,
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
            {
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.Tanh(),
                "num_layers": 2,
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
        ]

        poolings = [
            nn.MaxPool3d(kernel_size=2, ceil_mode=True) for _ in range(len(conv_blocks))
        ]
        reductional_layer = nn.Conv3d(128, 1, 1)  # Identity activation
        fully_connected = [256, 128, 64]

        super().__init__(
            input_size=input_size,
            n_modalities=2,
            conv_blocks=conv_blocks,
            poolings=poolings,
            reduction_layer=reductional_layer,
            fully_connected=fully_connected,
            n_classes=2,
            **kwargs
        )


# Just more flat features
class BimodalRimNetPlus(ModularRimNet):
    def __init__(self, input_size, batchnorm=False, **kwargs):
        conv_blocks = [
            {
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.Tanh(),
                "num_layers": 2,
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
            {
                "in_channels": 32,
                "out_channels": 64,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.Tanh(),
                "num_layers": 2,
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
            {
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.Tanh(),
                "num_layers": 2,
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
        ]

        poolings = [
            nn.MaxPool3d(kernel_size=2, ceil_mode=True) for _ in range(len(conv_blocks))
        ]
        reductional_layer = nn.Conv3d(128, 1, 1)  # Identity activation
        fully_connected = [512, 256, 128]

        super().__init__(
            input_size=input_size,
            n_modalities=2,
            conv_blocks=conv_blocks,
            poolings=poolings,
            reduction_layer=reductional_layer,
            fully_connected=fully_connected,
            n_classes=2,
            **kwargs
        )


class MonomodalRimNet(ModularRimNet):
    def __init__(self, input_size, batchnorm=False, **kwargs):
        conv_blocks = [
            {
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.Tanh(),
                "num_layers": 2,
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
            {
                "in_channels": 32,
                "out_channels": 64,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.Tanh(),
                "num_layers": 2,
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
            {
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.Tanh(),
                "num_layers": 2,
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
        ]

        poolings = [
            nn.MaxPool3d(kernel_size=2, ceil_mode=True) for _ in range(len(conv_blocks))
        ]
        reductional_layer = nn.Conv3d(128, 1, 1)  # Identity activation
        fully_connected = [128, 64]

        super().__init__(
            input_size=input_size,
            n_modalities=1,
            conv_blocks=conv_blocks,
            poolings=poolings,
            reduction_layer=reductional_layer,
            fully_connected=fully_connected,
            n_classes=2,
            **kwargs
        )


class TrimodalRimNet(ModularRimNet):
    def __init__(self, input_size, batchnorm=False, **kwargs):
        conv_blocks = [
            {
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.Tanh(),
                "num_layers": 2,
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
            {
                "in_channels": 32,
                "out_channels": 64,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.Tanh(),
                "num_layers": 2,
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
            {
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.Tanh(),
                "num_layers": 2,
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
        ]

        poolings = [
            nn.MaxPool3d(kernel_size=2, ceil_mode=True) for _ in range(len(conv_blocks))
        ]
        reductional_layer = nn.Conv3d(128, 1, 1)  # Identity activation
        fully_connected = [256, 128, 64]

        super().__init__(
            input_size=input_size,
            n_modalities=3,
            conv_blocks=conv_blocks,
            poolings=poolings,
            reduction_layer=reductional_layer,
            fully_connected=fully_connected,
            n_classes=2,
            **kwargs
        )


class TrimodalMaskRimNet(ModularRimNet):
    def __init__(self, input_size, batchnorm=False, **kwargs):
        # Define convolutional blocks for the first two modalities
        conv_blocks = [
            {
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.Tanh(),
                "num_layers": 2,  # First and second modality have 2 conv layers
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
            {
                "in_channels": 32,
                "out_channels": 64,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.Tanh(),
                "num_layers": 2,  # First and second modality have 2 conv layers
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
            {
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.Tanh(),
                "num_layers": 2,  # First and second modality have 2 conv layers
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
        ]
        
        # Customize the third modality with only 1 conv layer
        conv_blocks_modality_3 = [
            {
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.Tanh(),
                "num_layers": 2,  # Only 1 conv layer for the third modality
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
        ]

        # Pooling layers remain the same
        poolings = [
            nn.MaxPool3d(kernel_size=2, ceil_mode=True) for _ in range(len(conv_blocks))
        ]
        # Reduction layer
        reductional_layer = nn.Conv3d(128, 1, 1)  # Identity activation
        
        # Fully connected layers
        fully_connected = [256, 128, 64]

        # Call the parent class constructor
        super().__init__(
            input_size=input_size,
            n_modalities=3,
            conv_blocks=conv_blocks + conv_blocks + conv_blocks_modality_3,  # Custom conv blocks for the third modality
            poolings=poolings,
            reduction_layer=reductional_layer,
            fully_connected=fully_connected,
            n_classes=2,
            **kwargs
        )


class TrimodalSerendipityRimNet(ModularRimNet):
    def __init__(self, input_size, batchnorm=True, **kwargs):
        conv_blocks = [
            {
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.ReLU(),
                "num_layers": 3,
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
            {
                "in_channels": 32,
                "out_channels": 64,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.ReLU(),
                "num_layers": 3,
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
            {
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 3,
                "padding": 1,
                "activation": nn.ReLU(),
                "num_layers": 3,
                "use_batchnorm": batchnorm,
                "dropout_rate": 0,
            },
        ]
        poolings = [
            nn.MaxPool3d(kernel_size=2, ceil_mode=True) for _ in range(len(conv_blocks))
        ]
        reduction_layer = nn.Conv3d(128, 1, 1)
        fully_connected = [256, 128, 64]

        # Create an instance of your ModularRimNet model and move it to GPU
        super().__init__(
            input_size=input_size,
            n_modalities=3,
            conv_blocks=conv_blocks,
            poolings=poolings,
            reduction_layer=reduction_layer,
            fully_connected=fully_connected,
            n_classes=2,
            **kwargs
        )
