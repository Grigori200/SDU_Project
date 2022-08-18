from typing import *


import torch
from torch import nn


from models.base_tools import SAEBlock, InceptionBlock


class ResNetBlock(nn.Module):

    def __init__(
        self,
        channels: int,
        act_fn: Type[nn.Module] = nn.GELU,
        bias: bool = True,
        se: bool = True,
        **kwargs
    ) -> None:
        super(ResNetBlock, self).__init__()

        self.se = se
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=bias),
            nn.BatchNorm2d(channels),
            act_fn(),

            nn.Conv2d(channels, channels, 3, padding=1, bias=bias),
            nn.BatchNorm2d(channels)
        )

        if self.se:
            self.seb = SAEBlock(channels, act_fn=act_fn)

        self.act = act_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.conv(x)
        if self.seb:
            x2 = self.seb(x2)
        out = self.act_fn(x2 + x)
        return out



class ResNetInceptionBlock(nn.Module):
    
    def __init__(
        self,
        channels: int,
        act_fn: Type[nn.Module] = nn.GELU,
        se: bool = True,
        **kwargs
    ) -> None:
        super(ResNetInceptionBlock, self).__init__()

        self.se = se
        self.inception_block = InceptionBlock(channels, channels, act_fn)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        if self.se:
            self.seb = SAEBlock(channels, act_fn=act_fn)
        
        self.act_fn = act_fn()

    def forward(self, x):
        x2 = self.inception_block(x)
        x2 = self.conv2(x2)
        if self.seb:
            x2 = self.seb(x2)
        out = x + x2
        return self.act_fn(out)


RESNET_BLOCKS_DICT = {
    'resnet': ResNetBlock,
    'resnet_inception': ResNetInceptionBlock
}
