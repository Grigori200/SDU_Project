from typing import List

import torch
import numpy as np
from torch import nn



class EnsembleModel(nn.Module):

    def __init__(self, models: List[nn.Module]):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lgx = []
        for model in self.models:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            lgx.append(preds.cpu().item())
        preds = np.bincount(lgx).argmax()
        return preds
