from typing import *

import torch
from torch import nn
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import WandbLogger
from datamodules.datamodule import PneumoniaDataModule
from datamodules.transforms import train_transforms, test_val_transforms

from models import ResNet
from learning import train_test_model
from common import load_hyperparams
from adamp import AdamP


if __name__ == '__main__':
    """
    Main pipeline function.
    """
    # Load the hyperparameters.
    hyperparams = load_hyperparams()
    
    # Setup constant variables.
    WANDB_PROJECT_NAME = "PneumoniaSDU"
    ENTITY='kn-bmi'
    DATAMODULE_CLS = PneumoniaDataModule
    LOGS_DIR = "save"
    RUN_NAME = "TestRun"

    # Prepare DataModule hyperparameters.
    datamodule_kwargs = {
        "csv_path": "data_split_to_dirs.csv",
        "batch_size": 16,
        "size": 80,
        "normalize": True
    }

    # Prepare model hyperparameters.
    model_kwargs = {
        "in_channels": 1,
        "channels": 128,
        "out_channels": 128,
        "act_fn":  nn.GELU,
        "n_blocks":  4,
        "blocks_types": 'resnet_inception',
        "n_classes": 2,
        "dropout_pb":  0.5,            
    }
    
    # Prepare hyperparameters of objects used for training the model.
    trainer_kwargs = {
        'max_epochs': 100,

        "gpus": 1,

        "optimizer": AdamP,

        "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,

        "precision": 16,
        
        "strategy": "ddp",
    }
    
    # Prepare optimizer hyperparameters.
    optim_hparams = {
        "lr": 3e-4,
        "nesterov": True,
        "weight_decay": 0.05,
    }
    
    # Fixed label smoothing value.
    ls = 0.3 

    size = datamodule_kwargs["size"]
    
    # Create the datamodule with selected transformations.
    datamodule = PneumoniaDataModule(
        **datamodule_kwargs,
        train_transforms = train_transforms((size, size), datamodule_kwargs["normalize"]),
        val_transforms = test_val_transforms((size, size), datamodule_kwargs["normalize"]),
        test_transforms = test_val_transforms((size, size), datamodule_kwargs["normalize"]),
        )
    
    # Prepare the dataloaders.
    datamodule.prepare_data()
    
    # Create the model.
    model = ResNet(**model_kwargs)

    # Combine all hyperparamters.
    hparams = {
        "dataset": type(datamodule).__name__,
        **datamodule_kwargs,
        **model_kwargs,
        **trainer_kwargs,
        "train_size": len(datamodule.train_dataloader().dataset),
        "val_size": len(datamodule.val_dataloader().dataset),
        "test_size": len(datamodule.test_dataloader().dataset) 
    }

    # Create the logger.
    logger = WandbLogger(
        save_dir=str(LOGS_DIR),
        config=hparams,
        project=WANDB_PROJECT_NAME,
        log_model=False,
        entity=ENTITY
    )

    # Perform model training and evaluation.
    train_test_model(
        model=model,
        criterion=nn.CrossEntropyLoss(label_smoothing=ls),
        datamodule=datamodule,
        logger=logger,
        callbacks=[callbacks.EarlyStopping(monitor='val/acc', mode='max', min_delta=1e-4, patience=6)],
        **trainer_kwargs,
        optim_hparams=optim_hparams
    )
