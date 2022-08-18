from typing import *

import torch
from torch import nn
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import WandbLogger
from datamodules.datamodule import PneumoniaDataModule
from datamodules.dataset import PneumoniaData


from models import ResNet
from learning import train_test_model



if __name__ == '__main__':

    WANDB_PROJECT_NAME = "PneumoniaSDU"
    DATAMODULE_CLS = PneumoniaDataModule
    LOGS_DIR = "save"
    RUN_NAME = "TestRun"

    datamodule_kwargs = {
        "csv_path": "/home/konradkaranowski/SDU_Project/data_split_to_dirs.csv",
        "batch_size": 32,
    }

    model_kwargs = {
        "in_channels": 1,
        "channels": 64,
        "out_channels": 64,
        "act_fn":  nn.GELU,
        "n_blocks":  2,
        "blocks_types": 'resnet',
        "n_classes": 2,
        "dropout_pb":  0.3,            
    }
    trainer_kwargs = {
        'max_epochs': 100,

        "gpus": -1,

        "optimizer": torch.optim.AdamW,

        "optim_hparams": {
            "lr": 1e-4,
            "weight_decay": 1e-5
        },

        "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,

        "scheduler_hparams": {
        },

        "precision": 16,
        "strategy": "ddp",
    }

    datamodule = PneumoniaDataModule(**datamodule_kwargs)
    datamodule.prepare_data()
    model = ResNet(**model_kwargs)


    hparams = {
        "dataset": type(datamodule).__name__,
        **datamodule_kwargs,
        **model_kwargs,
        #**trainer_kwargs,
        "train_size": len(datamodule.train_dataloader().dataset),
        "val_size": len(datamodule.val_dataloader().dataset),
        "test_size": len(datamodule.test_dataloader().dataset) 
    }

    logger = WandbLogger(
        save_dir=str(LOGS_DIR),
        config=hparams,
        project=WANDB_PROJECT_NAME,
        log_model=False,
    )

    train_test_model(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        datamodule=datamodule,
        logger=logger,
        callbacks=[callbacks.EarlyStopping(monitor='val/loss', min_delta=1e-4, patience=3)],
        **trainer_kwargs
    )
