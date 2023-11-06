import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import Accuracy

from invertible_cl.nn.blocks import MLP
from invertible_cl.nn.functional import eval_mode


class Probing(pl.LightningModule):
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
        self.linear_head = nn.Linear(embed_dim, num_classes)
        self.nonlinear_head = MLP(embed_dim, embed_dim, num_classes)

        self.val_linear_acc = Accuracy('multiclass', num_classes=self.num_classes).to(self.device)
        self.val_nonlinear_acc = Accuracy('multiclass', num_classes=self.num_classes).to(self.device)
        self.test_linear_acc = Accuracy('multiclass', num_classes=self.num_classes).to(self.device)
        self.test_nonlinear_acc = Accuracy('multiclass', num_classes=self.num_classes).to(self.device)

        self.lr = lr
        self.lr_warmup_epochs = lr_warmup_epochs
        self.weight_decay = weight_decay

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        images, labels = batch

        with torch.no_grad(), eval_mode(self.encoder):
            embeds = self.encoder(images)

        for prefix in ['linear', 'nonlinear']:
            head = getattr(self, f'{prefix}_head')
            loss = F.cross_entropy(head(embeds), labels)
            self.log(f'train/{prefix}_probing_loss', loss, on_epoch=True)
            self.manual_backward(loss)

        optimizer.step()

    def on_validation_epoch_start(self) -> None:
        self.val_linear_acc.reset()
        self.val_nonlinear_acc.reset()

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        embeds = self.encoder(images)
        self.val_linear_acc.update(self.linear_head(embeds), labels)
        self.val_nonlinear_acc.update(self.nonlinear_head(embeds), labels)

    def on_validation_epoch_end(self) -> None:
        self.log(f'val/linear_probing_accuracy', self.val_linear_acc.compute())
        self.log(f'val/nonlinear_probing_accuracy', self.val_nonlinear_acc.compute())

    def on_test_start(self) -> None:
        self.test_linear_acc.reset()
        self.test_nonlinear_acc.reset()

    def test_step(self, batch, batch_idx):
        images, labels = batch
        embeds = self.encoder(images)
        self.test_linear_acc.update(self.linear_head(embeds), labels)
        self.test_nonlinear_acc.update(self.nonlinear_head(embeds), labels)

    def on_test_end(self) -> None:
        self.log(f'test/linear_probing_accuracy', self.test_linear_acc.compute())
        self.log(f'test/nonlinear_probing_accuracy', self.test_nonlinear_acc.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        assert self.trainer.max_epochs != -1
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.lr_warmup_epochs, max_epochs=self.trainer.max_epochs
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class OnlineProbing(pl.Callback):
    def __init__(
            self,
            embed_dim: int,
            num_classes: int,
            lr: float = 3e-4
    ) -> None:
        super().__init__()

        self.linear_head = nn.Linear(embed_dim, num_classes)
        self.linear_optimizer = torch.optim.Adam(self.linear_head.parameters(), lr=lr)

        self.nonlinear_head = MLP(embed_dim, embed_dim, num_classes)
        self.nonlinear_optimizer = torch.optim.Adam(self.nonlinear_head.parameters(), lr=lr)

        self.linear_acc = Accuracy('multiclass', num_classes=num_classes)
        self.nonlinear_acc = Accuracy('multiclass', num_classes=num_classes)

    def on_fit_start(self, trainer, pl_module):
        self.linear_head.to(pl_module.device)
        self.nonlinear_head.to(pl_module.device)

        self.linear_acc.to(pl_module.device)
        self.nonlinear_acc.to(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        (images, *_), labels = batch

        with torch.no_grad(), eval_mode(pl_module.encoder):
            embeds = pl_module.encoder(images)

        for prefix in ['linear', 'nonlinear']:
            head = getattr(self, f'{prefix}_head')
            optimizer = getattr(self, f'{prefix}_optimizer')

            optimizer.zero_grad()
            loss = F.cross_entropy(head(embeds), labels)
            pl_module.log(f'train/{prefix}_probing_loss', loss, on_epoch=True, sync_dist=True)
            loss.backward()
            optimizer.step()

    def on_validation_epoch_start(self, trainer, pl_module):
        self.linear_acc.reset()
        self.nonlinear_acc.reset()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        images, labels = batch

        with torch.no_grad(), eval_mode(pl_module.encoder):
            embeds = pl_module.encoder(images)

        self.linear_acc.update(self.linear_head(embeds), labels)
        self.nonlinear_acc.update(self.nonlinear_head(embeds), labels)

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.log(f'val/linear_probing_accuracy', self.linear_acc.compute(), sync_dist=True)
        pl_module.log(f'val/nonlinear_probing_accuracy', self.nonlinear_acc.compute(), sync_dist=True)
