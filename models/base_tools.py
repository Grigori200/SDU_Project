from typing import *


import torch
from torch import nn


# class ConvNormAct(nn.Module):

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel: Tuple[int, int] = (1, 1),
#         padding: int = 0,
#         act_fn: Type[nn.Module] = nn.GELU,
#         normalization: bool = True,
#         bias: bool = True
#         ) -> None:
#         super(ConvNormAct, self).__init__()
#         if normalization:
#             modules = [
#                 nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding, bias=bias),
#                 nn.BatchNorm2d(out_channels),
#                 act_fn()
#             ]
#         else:
#             modules = [
#                 nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding, bias=bias),
#                 act_fn()
#             ]
#         self.cnn = nn.Sequential(*modules)


#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.cnn(x)


# class SAEBlock(nn.Module):
    
#     def __init__(
#         self,
#         channels: int,
#         reduction: int = 16,
#         normalization: bool = False,
#         act_fn: Type[nn.Module] = nn.GELU
#     ) -> None:
#         super(SAEBlock, self).__init__()
#         assert channels >= reduction, f'Reduction = {reduction} must be <= than num of channels = {channels}'
#         self.cnn = nn.Sequential(
#             ConvNormAct(
#                 channels, 
#                 channels // reduction, 
#                 kernel=(1, 1), 
#                 bias=False, 
#                 normalization=False, 
#                 act_fn=act_fn
#             ),
#             nn.Conv2d(channels // reduction, channels * 2, 1, bias=False)
#         )
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         dx = nn.functional.adaptive_avg_pool2d(x, 1)
#         dx = self.cnn(dx)
#         dx1, dx2 = dx.split(dx.data.size(1) // 2, dim=1)
#         dx1 = torch.sigmoid(dx1)

#         return x * dx1 + dx2 


# class InceptionBlock(nn.Module):
    
#     def __init__(self,
#         input_channels: int,
#         channels: int,
#         act_fn: nn.Module
#         ) -> None:
#         super(InceptionBlock, self).__init__()
        
#         self.b1 = ConvNormAct(
#             in_channels=input_channels,
#             out_channels=channels,
#             kernel=1, 
#             normalization=True, 
#             act_fn=act_fn
#         )
        
#         self.b2 = nn.Sequential(
#                 ConvNormAct(
#                 in_channels=input_channels,
#                 out_channels=channels,
#                 kernel=1, 
#                 normalization=True, 
#                 act_fn=act_fn
#             ),
#                 ConvNormAct(
#                 in_channels=input_channels,
#                 out_channels=channels,
#                 kernel=(3, 3), 
#                 normalization=True, 
#                 act_fn=act_fn
#             )
#         )
        
#         self.b3 = nn.Sequential(
#             nn.Conv2d(input_channels, channels, (1, 1)),
#             nn.BatchNorm2d(channels),
#             act_fn(),
            
#             nn.Conv2d(channels, channels, (5, 5), padding=2),
#             nn.BatchNorm2d(channels),
#             act_fn() 
#         )
        
#         self.b4 = nn.Sequential(
#             nn.MaxPool2d((3, 3), 1, 1),
#                 ConvNormAct(
#                 in_channels=input_channels,
#                 out_channels=channels,
#                 kernel=1, 
#                 normalization=True, 
#                 act_fn=act_fn
#             )
#         )
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
        
#         x1 = self.b1(x)
#         x2 = self.b2(x)
#         x3 = self.b3(x)
#         x4 = self.b4(x)
        
#         cat = torch.cat([x1, x2, x3, x4], dim=1)
        
#         return cat



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
            #nn.ReLU(),
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