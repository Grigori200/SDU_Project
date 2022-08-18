
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
        lr_scheduler: torch.optim.lr_scheduler = None,
        optim_hparams: Dict[str, Any] = {'lr': 1e-4},
        scheduler_hparams: Dict[str, Any] = {'epochs': 5},

    ) -> None:
        super(Classifier, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.criterion = criterion
        
        # optimizer
        self.optimizer = optimizer
        self.optim_hparams = optim_hparams

        self.lr_scheduler = lr_scheduler,
        self.scheduler_hparams = scheduler_hparams

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()


    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        x, y = batch['x'], batch['y']
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
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
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_acc.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        self.test_acc.reset()

    def configure_optimizers(self) -> Tuple[
            List[torch.optim.Optimizer], 
            List[torch.optim.lr_scheduler._LRScheduler]
        ]:
        """
        Configure optimizer and returns it
        :return: torch optimizer
        """
        optimizer = self.optimizer(self.model.parameters(), **self.optim_hparams)
        scheduler = self.optimizer(optimizer, **self.scheduler_hparams)
        return [optimizer], [scheduler]

    def lr_scheduler_step(
        self, scheduler: torch.optim.lr_scheduler._LRScheduler, optimizer_idx, metric
    ) -> None:
        scheduler.step(epoch=self.current_epoch, metric=metric)
