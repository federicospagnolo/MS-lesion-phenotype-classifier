import inspect, sys

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor

from typing import Union, List, Tuple, Dict

from src.utils import validate_keys_in_dict, make_list


class ConvBlock(nn.Module):
    """
    Convolutional block module.

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        padding: Padding size.
        activation: Activation function.
        num_layers: Number of convolutional layers.
        use_batchnorm: Whether to use batch normalization.
        dropout_rate: Dropout rate.

    Methods:
        forward: Forward pass through the convolutional block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        padding: Union[int, List[int]],
        activation: Union[nn.Module, List[nn.Module]],
        num_layers: int,
        use_batchnorm: Union[bool, List[bool]] = True,
        dropout_rate: Union[float, List[float]] = 0.0,
    ) -> None:
        super(ConvBlock, self).__init__()

        # Store parameters as private attributes
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        
        # Transform parameters to lists if needed
        self.kernel_size = kernel_size
        # Due to the inner working of pytorch the classic setter way does not work with nn.modules
        # https://discuss.pytorch.org/t/python-property-getter-and-setter-with-nn-module/39521
        # Setter and getter arent use this way, so I comment then for now
        self.activation = self._make_list(activation, "activation")
        self.padding = padding
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        # print(self.kernel_size, self.padding, self.activation, self.use_batchnorm,self.dropout_rate)

        layers = nn.ModuleList()
        # self.conv_layer = nn.ModuleList()
        for i in range(num_layers):
            conv_layer = nn.Conv3d(
                in_channels, out_channels, self.kernel_size[i], padding=self.padding[i]
            )
            layers.append(conv_layer)
            if self.use_batchnorm[i]:
                layers.append(nn.BatchNorm3d(self.out_channels))
            # BN before activation so the canocical way but never is clear where is better
            layers.append(self.activation[i])

            if self.dropout_rate[i] > 0.0:
                layers.append(nn.Dropout3d(p=self.dropout_rate[i]))

            in_channels = out_channels

        # Create the convolutional block using Sequential
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)

    def _make_list(self, param, name: str) -> List:
        return make_list(param, self.num_layers, name)

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, kernel_size):
        self._kernel_size = self._make_list(kernel_size, "kernel size")

    @property
    def padding(self) -> Union[int, List[int]]:
        return self._padding

    @padding.setter
    def padding(self, value: Union[int, List[int]]) -> None:
        self._padding = self._make_list(value, "padding")

    @property
    def use_batchnorm(self) -> Union[bool, List[bool]]:
        return self._use_batchnorm

    @use_batchnorm.setter
    def use_batchnorm(self, value: Union[bool, List[bool]]) -> None:
        self._use_batchnorm = self._make_list(value, "use_batchnorm")

    @property
    def dropout_rate(self) -> Union[float, List[float]]:
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, value: Union[float, List[float]]) -> None:
        self._dropout_rate = self._make_list(value, "dropout_rate")

    @staticmethod
    def parameters():
        return inspect.signature(ConvBlock.__init__).parameters


class ModularRimNet(nn.Module):
    """
    Modular RimNet module.
    Features are extracted from each modality. The ones extracted form the first modality (FLAIR in the original RimNet)
    convolution block are concataneated to the rest of employed modalities.
    Following the RimNet implemtation the convolutional block are the same for each modality to be employed

    Attributes:
        input_size: Size of the input volume (W, H, D).
        n_modalities: Number of input modalities.
        conv_blocks: List of convolutional blocks for each modality.
        poolings: List of pooling layers.
        reduction_layer: Feature aggregation layer at the end of each modality forward.
        fully_connected: List of dimensions for fully connected layers.
        n_classes: Number of output classes.
        xavier_init: Whether to use Xavier initialization as in previous RimNet.

    Methods:
        forward: Forward pass through the ModularRimNet.
        Expects to recieve a tensor with shape [B, n_modalities, W, H, D]
    """

    def __init__(
        self,
        input_size: Tuple[int, int, int],
        n_modalities: int,
        conv_blocks: List[dict],
        poolings: List[nn.Module],
        reduction_layer: nn.Module,
        fully_connected: List[int],
        n_classes: int,
        xavier_init=True,  # From classic RimNet
    ) -> None:
        super(ModularRimNet, self).__init__()
        assert len(input_size) == 3, "the input size length must be 3"
        assert n_classes > 0, "The number of classes must be at least one"
        assert isinstance(fully_connected, list)
        assert len(conv_blocks) > 1

        self.input_size = input_size
        self.n_modalities = n_modalities
        self.conv_blocks = conv_blocks
        self.n_conv_blocks = len(conv_blocks)
        self.n_classes = n_classes
        # getters and setter with nn.modules are tricky. check ConvBlock for activation
        self.poolings = make_list(poolings, self.n_conv_blocks, "poolings")
        self.reduction_layer = make_list(
            reduction_layer, self.n_modalities, "reduction layers"
        )

        # I always extract features to be fed to the rest of modalities network (subnets) from the first one (Following the paper, the FLAIR)
        # just the convblock follow by the corresponding pool
        self.subnet_for_modality = nn.ModuleList()

        # First modality (only using the first ConvBlock)
        self.subnet_for_modality.append(
            nn.ModuleList(
                [
                    nn.Sequential(ConvBlock(**self.conv_blocks[0]), self.poolings[0])
                ]
            )
        )

        # Create aux_conv_blocks for all modalities (including the second and third modalities)
        aux_conv_blocks = [cv.copy() for cv in self.conv_blocks]

        # For the forward concat, we need to double the input channels for the second ConvBlock
        if self.n_conv_blocks > 1: #comment this for ablation study!!!!!!!!!
            aux_conv_blocks[1].update(
                {"in_channels": aux_conv_blocks[1]["in_channels"] * 2}
            )

        # Second and third modalities (using all ConvBlocks from aux_conv_blocks)
        for subnet in range(1, self.n_modalities):  # assuming n_modalities includes all three modalities
            self.subnet_for_modality.append(
                nn.ModuleList(
                    [
                        nn.Sequential(ConvBlock(**cv), self.poolings[p])
                        for p, cv in enumerate(aux_conv_blocks)
                    ]
                )
            )
            

        # adding reduction layer
        for subnet, reductional_layer in enumerate(self.reduction_layer):
            if subnet != 0:
                self.subnet_for_modality[subnet].append(reductional_layer)

        # I need the number of features after all convolution for fullyconnected layers definition
        (self.convolutional_features_shape, self.n_extracted_features) = self._infer_subnet_output()

        # Check the number of features per modality
        #print("Extracted features per modality:", self.n_extracted_features)

        # Adjust for the number of modalities
        self.n_extracted_features *= self.n_modalities  # Total features from all modalities

        # Verify the adjusted number of extracted features
        #print("Total extracted features after modalities:", self.n_extracted_features)

        # Define the dimensions for fully connected layers
        fully_connected_dims = [self.n_extracted_features] + fully_connected + [self.n_classes]
        fully_connected_dims = fully_connected + [self.n_classes]

        # Print the dimensions for debugging
        #print("Fully connected dimensions:", fully_connected_dims)

        # Create the fully connected layer sequentially
        self.fcl = nn.Sequential(
            *[nn.Linear(fully_connected_dims[fc], fully_connected_dims[fc + 1]) 
              for fc in range(len(fully_connected_dims) - 1)]
        )

        if xavier_init:
            self.initialize_layers()

    @property
    def conv_blocks(self) -> List[dict]:
        return self._conv_blocks

    @conv_blocks.setter
    def conv_blocks(self, conv_blocks) -> None:
        assert isinstance(
            conv_blocks, list
        ), "conv_blocks must be a List[dict(**ConvBlock_params)]"
        mandatory_params = [
            name
            for name, param in ConvBlock.parameters().items()
            if param.default == inspect.Parameter.empty and name != "self"
        ]
        for dct in conv_blocks:
            validate_keys_in_dict(dct, mandatory_params)
        # if doesnt break
        self._conv_blocks = conv_blocks

    def _infer_subnet_output(self):
        with torch.no_grad():
            to_forward_fake = nn.Sequential(*self.subnet_for_modality[0])
            output_fake = to_forward_fake(torch.ones(1, 1, *self.input_size))
            return output_fake.shape, torch.numel(output_fake)

    # These initialization mimics the one from original RimNet. Not necesarily the best and for sure is not SOTA
    # The initialization methods by default follow the defualt give by the tflearn library used by RimNet within tensorflow
    # tflearn employed was 0.3.2 --> https://github.com/tflearn/tflearn/blob/0.3.2/tflearn/layers/conv.py
    # Please extend the class to use new different initialitations
    def initialize_layers(self):
        def convblock_init(convblock: ConvBlock):
            for cb, layer in enumerate(convblock.conv_block):
                if isinstance(layer, nn.Conv3d):
                    for param_name, param in layer.named_parameters():
                        if "weight" in param_name:
                            init.xavier_uniform_(param)
                        elif "bias" in param_name:
                            init.zeros_(param)

        for s, subnet in enumerate(
            self.subnet_for_modality
        ):  # subnet[Sequential(convblock0,pooling0), Sequential(convblock1,pooling1),...,redutional_layer]
            for b, block in enumerate(subnet):
                if isinstance(block, nn.Sequential):
                    for sb, seq_block in enumerate(block):
                        if isinstance(seq_block, ConvBlock):
                            convblock_init(
                                seq_block
                            )  # The other choice is pooling and there is not init defined
                elif isinstance(block, nn.Conv3d):
                    for param_name, param in block.named_parameters():
                        if "weight" in param_name:
                            init.trunc_normal_(
                                param, std=0.02
                            )  # https://github.com/tflearn/tflearn/blob/0.3.2/tflearn/initializations.py
                        elif "bias" in param_name:
                            init.zeros_(param)
        # https://github.com/tflearn/tflearn/blob/0.3.2/tflearn/layers/core.py
        for fc in self.fcl:
            for param_name, param in fc.named_parameters():
                if "weight" in param_name:
                    init.trunc_normal_(param, std=0.02)
                elif "bias" in param_name:
                    init.zeros_(param)

    def forward(self, x):
        # Split the input tensor x by the number of modalities
        # Assume htat x concat modalities in the Channels dimeshon (dim=1)
        x_splits = torch.split(x, 1, dim=1)
        # print("x_splits", len(x_splits), x_splits[0].shape)

        # Forward pass through each subnet
        subnet_outputs = []
        for subnet_idx, subnet in enumerate(self.subnet_for_modality):
            subnet_input = x_splits[subnet_idx]
            for block_idx, block in enumerate(subnet):
                subnet_input = block(subnet_input)
                if subnet_idx == 0 and block_idx == 0:
                    first_modality_and_convblock_output = (
                        subnet_input.clone()
                    )  # convblock+pooling actually
                elif subnet_idx == 2 and block_idx == 0:
                    subnet_input = torch.cat(
                        (first_modality_and_convblock_output, subnet_input), dim=1
                    )
                # Comment for ablation study!!!!!!!!!!!!
                elif (
                    subnet_idx == 1 and block_idx == 0
                ):  # so concat the output of first conblock from first modality to the rest
                    subnet_input = torch.cat(
                        (first_modality_and_convblock_output, subnet_input), dim=1
                    )
                #print(block_idx, subnet_idx, subnet_input.size())   
            #print("input")
            #print(subnet_input.size())
            if subnet_idx != 0:
                 subnet_outputs.append(subnet_input)
            #print("outputs")
            #for x in subnet_outputs:
                 #print(x.size())

        # Concatenate the outputs from the first block of each subnet along the channel dimension
        concatenated_output = torch.cat(subnet_outputs, dim=1)

        # Flatten the concatenated output
        flattened_output = concatenated_output.view(concatenated_output.size(0), -1)

        # Forward pass through fully connected layers
        final_output = self.fcl(flattened_output)
        # print("FINAL OUTPUT", final_output)

        return final_output

    def __str__(self):
        result = ""
        for modality in range(self.n_modalities):
            result += f"----- Modality: {modality} ----------\n"
            for layer in self.subnet_for_modality[modality]:
                result += f"{layer}\n"
        for i, fc in enumerate(self.fcl):
            result += f"FC {i}: {fc}\n"
        return result
