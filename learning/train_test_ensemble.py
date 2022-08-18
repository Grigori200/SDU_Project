from typing import *

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.callbacks import Callback
from torch import nn

from torchensemble import VotingClassifier
from torchensemble.utils.logging import set_logger


def train_test_model_ensemble(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: str, 
    lr_scheduler: str,
    optim_hparams: Dict,
    scheduler_hparams: Dict,
    datamodule: pl.LightningDataModule,
    max_epochs: int = 100,
    n_models: int = 10
):
    """
    Perform training or testing loop for a given model.

    Args:
        model (nn.Module): a model to train or test.
        criterion (nn.Module): a criterion function.
        optimizer (str): an optimizer.
        lr_scheduler (str): a learning rate scheduler.
        optim_hparams (Dict): optimizer hyperparameters.
        scheduler_hparams (Dict): learning rate scheduler hyperparameters.
        datamodule (pl.LightningDataModule): a DataModule providing DataLoaders.
        max_epochs (int, optional): a maximum numer of epochs. Defaults to 100.
    """
    module = VotingClassifier(
        estimator=model,
        n_estimators=n_models,
        cuda=False
    )
    
    module.set_criterion(criterion)
    module.set_optimizer(optimizer,
                         lr=optim_hparams['lr'],
                         weight_decay=optim_hparams['weight_decay'])
    module.set_scheduler(lr_scheduler,
                         T_max=scheduler_hparams['T_max'])
    
    logger = set_logger(f"ensemble_{optimizer}_decay{optim_hparams['weight_decay']}_epochs{max_epochs}_models{n_models}")
    
    module.fit(
        datamodule.train_dataloader(),
        epochs=max_epochs,
        test_loader=datamodule.val_dataloader()
    )
    acc = module.evaluate(datamodule.test_dataloader())
    print(f"Test acc: {acc}")
    