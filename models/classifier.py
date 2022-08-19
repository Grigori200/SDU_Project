from typing import *

import torch
import pytorch_lightning as pl
from torch import nn
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy


class Classifier(pl.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr_scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
        optim_hparams={},
        **kwargs

    ) -> None:
        """
        A pytorch lightning model wrapper performing data and operations flow.

        Args:
            model (nn.Module): The model used for classification.
            criterion (nn.Module): The criterion function.
            optimizer (torch.optim.Optimizer, optional): The optimizer. Defaults to torch.optim.Adam.
            lr_scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler. Defaults to torch.optim.lr_scheduler.ReduceLROnPlateau.
            optim_hparams (dict, optional): The optimizer hyperparameters. Defaults to {}.
        
        Author: Adam
        """
        super(Classifier, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.criterion = criterion
        
        # optimizer
        self.optimizer = optimizer
        self.optim_hparams = optim_hparams

        self.lr_scheduler = lr_scheduler,

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()


    def forward(self, x: torch.Tensor):
        """
        Fowards the input through the model.

        Args:
            x (torch.Tensor): an input tensor.

        Returns:
            torch.Tensor: an output tensor processed through model.
            
        Author: Adam
        """
        return self.model(x)

    def step(self, batch: torch.Tensor):
        """
        Performs one step of classification.

        Args:
            batch (torch.Tensor): the batch of data.

        Returns:
            Tuple[nn.Module, torch.Tensor, torch.Tensor]: the tuple of loss, predicted labels and true labels.
        
        Author: Adam
        """
        x, y = batch['x'], batch['y'].long()
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        """
        Performs training step and logs metrics.

        Args:
            batch (Any): the batch of data.
            batch_idx (int): the batch_idx, needed for compatibility with framework.

        Returns:
            Dict[str, torch.Tensor]: a dictionary containing the training loss, predicted labels and true labels. 
        
        Author: Adam
        """
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        """
        Provides operation to perform at the end of training epoch.

        Args:
            outputs (List[Any]): a list of dicts returned from training_step().
        
        Author: Adam
        """
        self.train_acc.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        """
        Performs validation step and logs metrics.

        Args:
            batch (Any): the batch of data.
            batch_idx (int): the batch_idx, needed for compatibility with framework.

        Returns:
            Dict[str, torch.Tensor]: a dictionary containing the validation loss, predicted labels and true labels. 
        
        Author: Adam
        """
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        """
        Provides operation to perform at the end of validation epoch.
        Computes accuracy for current epoch, and performs logging.

        Args:
            outputs (List[Any]): a list of dicts returned from validation_step().
        
        Author: Adam
        """
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch: Any, batch_idx: int):
        """
        Performs testing step and logs metrics.

        Args:
            batch (Any): the batch of data.
            batch_idx (int): the batch_idx, needed for compatibility with framework.

        Returns:
            Dict[str, torch.Tensor]: a dictionary containing the test loss, predicted labels and true labels. 
        
        Author: Adam
        """
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        """
        Provides operation to perform at the end of testing epoch.

        Args:
            outputs (List[Any]): a list of dicts returned from test_step().
        
        Author: Adam
        """
        self.test_acc.reset()

    def configure_optimizers(self):
        """
        Configure optimizer and return it
        
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR] a tuple of optimizer and learning rate scheduler used for training. 
        
        Author: Adam
        """
        optimizer = self.optimizer(self.model.parameters(), lr=self.optim_hparams["lr"], 
            weight_decay=self.optim_hparams["weight_decay"], nesterov=self.optim_hparams["nesterov"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100) 
        return [optimizer], [scheduler]

    def lr_scheduler_step(
        self, scheduler: torch.optim.lr_scheduler._LRScheduler, optimizer_idx: int, metric: Optional[Any]
    ) -> None:
        """
        Perform the learning rate scheduler step.

        Args:
            scheduler (torch.optim.lr_scheduler._LRScheduler): the learning rate scheduler used in the training process.
            optimizer_idx (int): Index of the optimizer associated with this scheduler.
            metric (Optional[Any]): Value of the monitor used for ReduceLROnPlateau learning rate scheduler.
        
        Author: Adam
        """
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch=self.current_epoch, metrics=metric)
        else:
            scheduler.step(epoch=self.current_epoch)
