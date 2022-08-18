from typing import *

import torch
import pytorch_lightning as pl
from torch import nn

from models import Classifier


def train_test(
    model: nn.Module,
    criterion,
    optimizer, 
    lr_scheduler,
    optim_hparams,
    scheduler_hparams,
    gpus,
    logger,
    callbacks,
    datamodule,
    max_epochs: int = 100

):
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
        logger=logger
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
