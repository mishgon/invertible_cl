from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import Accuracy


class EndToEnd(pl.LightningModule):
    def __init__(
            self,
            encoder: nn.Module,
            embed_dim: int,
            num_classes: int,
            lr: float = 1e-2,
            lr_warmup_epochs: int = 10,
            weight_decay: float = 1e-6,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore='encoder')

        self.encoder = encoder
        self.head = nn.Linear(embed_dim, num_classes)

        self.val_acc = Accuracy('multiclass', num_classes=num_classes)
        self.test_acc = Accuracy('multiclass', num_classes=num_classes)

        self.lr = lr
        self.lr_warmup_epochs = lr_warmup_epochs
        self.weight_decay = weight_decay

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        loss = F.cross_entropy(self.head(self.encoder(images)), labels)
        self.log('train/loss', loss, on_epoch=True, on_step=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_acc.reset()

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        self.val_acc.update(self.head(self.encoder(images)), labels)

    def on_validation_epoch_end(self) -> None:
        self.log(f'val/accuracy', self.val_acc.compute())

    def on_test_start(self) -> None:
        self.test_acc.reset()

    def test_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        self.test_acc.update(self.head(self.encoder(images)), labels)

    def on_test_end(self) -> None:
        self.log(f'test/accuracy', self.test_acc.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        assert self.trainer.max_epochs != -1
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.lr_warmup_epochs, max_epochs=self.trainer.max_epochs
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
