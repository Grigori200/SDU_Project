from typing import *

import torch
from torch import nn
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import WandbLogger
from datamodules.datamodule import PneumoniaDataModule
from datamodules.dataset import PneumoniaData
from adamp import AdamP


from models import XGBoostCNN
from models.xgboost_cnn import train_xgboost
from learning import train_test_model
from common import load_hyperparams


if __name__ == '__main__':

    WANDB_PROJECT_NAME = "PneumoniaSDU"
    DATAMODULE_CLS = PneumoniaDataModule
    LOGS_DIR = "save"
    RUN_NAME = "TestRun"

    datamodule_kwargs = {
        "csv_path": "data_split_to_dirs.csv",
        "batch_size": 32,
    }

    model_kwargs = {
        "width": 224,
        "height": 224,
        "in_channels": 1,
        "num_classes": 2,
    }
    trainer_kwargs = {
        'max_epochs': 100,

        "gpus": -1,

        "optimizer": AdamP,

        "optim_hparams": {
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "nesterov": False,
        },

        "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,

        "precision": 16,
    }

    datamodule = PneumoniaDataModule(**datamodule_kwargs)
    datamodule.prepare_data()
    model = XGBoostCNN(**model_kwargs)


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

    train_xgboost(model, )

