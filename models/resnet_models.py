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
        **kwargs
    ) -> None:
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

        self.fc = nn.Linear(out_channels, n_classes)
        
    def forward(self, x):
        x = self.input_conv(x)
        x = self.blocks(x)
        x = self.output_conv(x)

        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.data.size(0), -1)
        x = self.fc(x)
        return x
