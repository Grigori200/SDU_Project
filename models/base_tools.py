from typing import *


import torch
from torch import nn


class SAEBlock(nn.Module):
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        act_fn: Type[nn.Module] = nn.GELU
    ) -> None:
        super(SAEBlock, self).__init__()
        assert channels >= reduction, f'Reduction = {reduction} must be <= than num of channels = {channels}'
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            act_fn(),
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = nn.functional.adaptive_avg_pool2d(x, 1)
        dx = self.cnn(dx)
        dx1, dx2 = dx.split(dx.data.size(1) // 2, dim=1)
        dx1 = torch.sigmoid(dx1)

        return x * dx1 + dx2 


class InceptionBlock(nn.Module):
    
    def __init__(self,
        input_channels: int,
        channels: int,
        act_fn: nn.Module
        ) -> None:
        super(InceptionBlock, self).__init__()
        
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, channels, (1,1)),
            nn.BatchNorm2d(channels),
            act_fn()
        )
        
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, channels, (1, 1)),
            nn.BatchNorm2d(channels),
            act_fn(),
            
            nn.Conv2d(channels, channels, (3, 3), padding=1),
            nn.BatchNorm2d(channels),
            act_fn()
        )
        
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, channels, (1, 1)),
            nn.BatchNorm2d(channels),
            act_fn(),
            
            nn.Conv2d(channels, channels, (5, 5), padding=2),
            nn.BatchNorm2d(channels),
            act_fn() 
        )
        
        self.b4 = nn.Sequential(
            nn.MaxPool2d((3, 3), 1, 1),
            nn.Conv2d(input_channels, channels, (1, 1)),
            nn.BatchNorm2d(channels),
            act_fn(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)
        x4 = self.b4(x)
        
        cat = torch.cat([x1, x2, x3, x4], dim=1)
        
        return cat