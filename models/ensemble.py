from typing import List

import torch
import numpy as np
from torch import nn



class EnsembleModel(nn.Module):

    def __init__(self, models: List[nn.Module]):
        """
        Creates an ensemble of provided models.

        Args:
            models (List[nn.Module]): an iterable of models.
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fowards the input through the ensemble of models.

        Args:
            x (torch.Tensor): an input tensor.

        Returns:
            torch.Tensor: an output tensor processed the ensemble of models with labels determined by majority voting.
        """
        lgx = []
        for model in self.models:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            lgx.append(preds.cpu().item())
        preds = np.bincount(lgx).argmax()
        return preds
