from typing import *

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch import nn

from models import Classifier


def train_test_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer, 
    lr_scheduler: object,
    optim_hparams: Dict,
    scheduler_hparams: Dict,
    gpus: int,
    logger,
    callbacks: Callback,
    datamodule: pl.LightningDataModule,
    max_epochs: int = 100,
    precision: int = 16,
    strategy: str = "ddp",
):
    """
    Perform training or testing loop for a given model.

    Args:
        model (nn.Module): a model to train or test.
        criterion (nn.Module): a criterion function.
        optimizer (torch.optim.Optimizer): an optimizer.
        lr_scheduler (object): a learning rate scheduler.
        optim_hparams (Dict): optimizer hyperparameters.
        scheduler_hparams (Dict): learning rate scheduler hyperparameters.
        gpus (int): a number of gpus to perform computations on.
        logger (Logger): a logger to perform logging with.
        callbacks (Callback): an iterable of lightning callbacks.
        datamodule (pl.LightningDataModule): a DataModule providing DataLoaders.
        max_epochs (int, optional): a maximum numer of epochs. Defaults to 100.
        precision (int, optional): a float precision in classifier training process. Defaults to 16.
        strategy (str, optional): a data distribution strategy. Defaults to "ddp".
    """
    module = Classifier(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        optim_hparams=optim_hparams,
        scheduler_hparams=scheduler_hparams,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=gpus,
        callbacks=callbacks,
        logger=logger,
        precision=precision,
        strategy=strategy,
    )
    trainer.fit(
        model=module,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader()
    )
    trainer.test(
        dataloaders=[datamodule.test_dataloader()]
    )
    logger.experiment.finish()