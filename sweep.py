from typing import *

import torch
from torch import nn
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import WandbLogger
from datamodules.datamodule import PneumoniaDataModule
from datamodules.dataset import PneumoniaData


from models import ResNet
from learning import train_test_model
from common import load_hyperparams
from adamp import AdamP


if __name__ == '__main__':
    hyperparams = load_hyperparams()
    WANDB_PROJECT_NAME = "PneumoniaSDU"
    DATAMODULE_CLS = PneumoniaDataModule
    LOGS_DIR = "save"
    RUN_NAME = "TestRun"

    datamodule_kwargs = {
        "csv_path": "/home/konradkaranowski/SDU_Project/data_split_to_dirs.csv",
        "batch_size": hyperparams["batch_size"],
    }

    model_kwargs = {
        "in_channels": 1,
        "channels": hyperparams["channels"],
        "out_channels": hyperparams["out_channels"],
        "act_fn":  nn.GELU,
        "n_blocks":  hyperparams["n_blocks"],
        "blocks_types": hyperparams["block_types"],
        "n_classes": 2,
        "dropout_pb":  hyperparams["dropout_pb"],            
    }
    trainer_kwargs = {
        'max_epochs': 100,

        "gpus": 1,

        "optimizer": AdamP,


        "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,


        "precision": 16,
        "strategy": "ddp",
    }
    optim_hparams = {
        "lr": hyperparams["learning_rate"],
        "nesterov": hyperparams["nesterov"],
        "weight_decay": hyperparams["weight_decay"],
    }

    datamodule = PneumoniaDataModule(**datamodule_kwargs)
    datamodule.prepare_data()
    model = ResNet(**model_kwargs)


    hparams = {
        "dataset": type(datamodule).__name__,
        **datamodule_kwargs,
        **model_kwargs,
        **trainer_kwargs,
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
        criterion=nn.CrossEntropyLoss(label_smoothing=hyperparams["label_smoothing"]),
        datamodule=datamodule,
        logger=logger,
        callbacks=[callbacks.EarlyStopping(monitor='val/loss', min_delta=1e-4, patience=3)],
        **trainer_kwargs,
        optim_hparams=optim_hparams
    )
