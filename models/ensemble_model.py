from typing import *

import torch
from torch import nn
from functorch import combine_state_for_ensemble, vmap

from resnet_models import ResNet


class EnsembleResNets(nn.Module): 
    def __init__(
        self,
        n_models: int,
        in_channels: int,
        channels: int,
        out_channels: int,
        act_fn: Type[nn.Module] = nn.GELU,
        n_blocks: int = 10,
        blocks_types: str = 'resnet',
        n_classes: int = 2,
        dropout_pb: float = 0.0,
        **kwargs
    ):
        self.models = [ResNet(
            in_channels,
            channels, 
            out_channels, 
            act_fn,
            n_blocks, 
            blocks_types, 
            n_classes, 
            dropout_pb,
            **kwargs
        ) for _ in range(n_models)]
        self.fmodel, self.params, self.buffers = combine_state_for_ensemble(self.models)

    def forward(self, x):
        preds = vmap(self.fmodel, in_dims=(0, 0, None))(self.params, self.buffers, x)
        # [n_models, batch_size, n_classes]
        print(preds.size())
        raise Exception()
        return preds
        