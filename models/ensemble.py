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
            
        Author: Konrad
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> np.ndarray:
        """
        Fowards the input through the ensemble of models.

        Args:
            x (torch.Tensor): an input tensor.

        Returns:
            np.ndarray: an output tensor processed the ensemble of models with labels determined by majority voting.
        
        Author: Konrad
        """
        lgx = []
        for model in self.models:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            lgx.append(preds.cpu().item())
        preds = np.bincount(lgx).argmax()
        return preds
