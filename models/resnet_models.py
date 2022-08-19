from typing import *


import torch
from torch import nn


from models.resnet_blocks import RESNET_BLOCKS_DICT


class ResNet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        channels: int,
        out_channels: int,
        act_fn: Type[nn.Module] = nn.GELU,
        n_blocks: int = 10,
        blocks_types: str = 'resnet',
        n_classes: int = 2,
        dropout_pb: float = 0.0,
        **kwargs
    ) -> None:
        """
        Convolutional Neural Network model with ResNet building blocks.

        Args:
            in_channels (int): The number of input channels.
            channels (int): The number of output channels before output convolution.
            out_channels (int): The number of output channels from feature extractor.
            act_fn (Type[nn.Module], optional): The activation function. Defaults to nn.GELU.
            n_blocks (int, optional): The number of ResNet blocks. Defaults to 10.
            blocks_types (str, optional): the type of ResNet blocks used to build the model. Defaults to 'resnet'.
            n_classes (int, optional): The number of output classes. Defaults to 2.
            dropout_pb (float, optional): The dropout probability. Defaults to 0.0.
        
        Author: Konrad
        """
        super(ResNet, self).__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, 2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            act_fn()
        )

        self.blocks = nn.Sequential(
            *[
                RESNET_BLOCKS_DICT[blocks_types](
                    channels=channels,
                    act_fn=act_fn,
                    **kwargs
                ) for _ in range(n_blocks - 1)
            ]
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(channels, out_channels, 2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act_fn()
        )

        self.dp = nn.Dropout(dropout_pb)
        self.fc = nn.Linear(out_channels, n_classes)
        
    def forward(self, x):
        """
        Fowards the input through the network.

        Args:
            x (torch.Tensor): an input tensor.

        Returns:
            torch.Tensor: an output tensor processed through network.
        
        Author: Konrad
        """
        x = self.input_conv(x)
        x = self.blocks(x)
        x = self.output_conv(x)

        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.data.size(0), -1)
        x = self.dp(x)
        x = self.fc(x)
        return x
