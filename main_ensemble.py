from typing import *

import torch
from torch import nn
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import WandbLogger
from datamodules.datamodule import PneumoniaDataModule
from datamodules.dataset import PneumoniaData


from models import ResNet
from learning import train_test_model_ensemble



if __name__ == '__main__':

    WANDB_PROJECT_NAME = "PneumoniaSDU"
    DATAMODULE_CLS = PneumoniaDataModule
    LOGS_DIR = "save"
    RUN_NAME = "TestRun"

    datamodule_kwargs = {
        "csv_path": "data_split_to_dirs.csv",
        "batch_size": 16,
    }

    model_kwargs = {
        "in_channels": 1,
        "channels": 64,
        "out_channels": 64,
        "act_fn":  nn.GELU,
        "n_blocks":  3,
        "blocks_types": 'resnet',
        "n_classes": 2,
        "dropout_pb":  0.3,            
    }
    trainer_kwargs = {
        'max_epochs': 20,

        "optimizer": 'AdamW',

        "optim_hparams": {
            "lr": 1e-4,
            "weight_decay": 1e-5
        },

        "lr_scheduler": 'CosineAnnealingLR',

        "scheduler_hparams": {
            'T_max': 100
        },
        
        "n_models": 9,
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

    train_test_model_ensemble(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        datamodule=datamodule,
        **trainer_kwargs
    )
