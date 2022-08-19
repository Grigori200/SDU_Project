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
        """
        A Squeeze and Excitation convolutional neural network building block.

        Args:
            channels (int): The number of input channels.
            reduction (int, optional): A coefficient of channels number reduction. Defaults to 16.
            act_fn (Type[nn.Module], optional): An activation function. Defaults to nn.GELU.
        """
        super(SAEBlock, self).__init__()
        assert channels >= reduction, f'Reduction = {reduction} must be <= than num of channels = {channels}'
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            act_fn(),
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fowards the input through the network.

        Args:
            x (torch.Tensor): an input tensor.

        Returns:
            torch.Tensor: an output tensor processed through network.
        """
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
        """
        A convolutional neural network building block introduced in Inception CNN family.

        Args:
            input_channels (int): The number of input channels.
            channels (int): The number of output channels in each of four branches of inception block. Total output number of channels equals to channels * 4.
            act_fn (nn.Module): An activation function.
        """
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
        """
        Fowards the input through the network.

        Args:
            x (torch.Tensor): an input tensor.

        Returns:
            torch.Tensor: an output tensor processed through network.
        """
        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)
        x4 = self.b4(x)
        
        cat = torch.cat([x1, x2, x3, x4], dim=1)
        
        return cat