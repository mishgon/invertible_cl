from typing import Any

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from invertible_cl.nn.encoder import encoder, EncoderArchitecture
from invertible_cl.nn.blocks import MLP


class SimCLR(pl.LightningModule):
    def __init__(
            self,
            encoder_architecture: EncoderArchitecture,
            proj_dim: int = 128,
            temp: float = 0.1,
            lr: float = 1e-2,
            weight_decay: float = 1e-6,
            warmup_epochs: int = 100,
            **encoder_kwargs: Any
    ) -> None:
        super().__init__()
        
        self.save_hyperparameters()

        self.encoder, self.embed_dim = encoder(encoder_architecture, **encoder_kwargs)
        self.projector = MLP(self.embed_dim, self.embed_dim, proj_dim, num_hidden_layers=2, bias=False)

        self.temp = temp
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

    def forward(self, images):
        return self.encoder(images)

    def training_step(self, batch, batch_idx):
        (_, views_1, views_2), _ = batch

        embeds_1 = F.normalize(self.projector(self(views_1)), dim=1)  # (batch_size, proj_dim)
        embeds_2 = F.normalize(self.projector(self(views_2)), dim=1)  # (batch_size, proj_dim)

        logits_11 = torch.matmul(embeds_1, embeds_1.T) / self.temp  # (batch_size, batch_size)
        logits_11.fill_diagonal_(float('-inf'))
        logits_12 = torch.matmul(embeds_1, embeds_2.T) / self.temp
        pos_logits = logits_12.diag()
        logits_22 = torch.matmul(embeds_2, embeds_2.T) / self.temp
        logits_22.fill_diagonal_(float('-inf'))

        loss_1 = torch.mean(-pos_logits + torch.logsumexp(torch.cat([logits_11, logits_12], dim=1), dim=1))
        loss_2 = torch.mean(-pos_logits + torch.logsumexp(torch.cat([logits_12.T, logits_22], dim=1), dim=1))
        loss = (loss_1 + loss_2) / 2
        self.log(f'pretrain/info_nce_loss', loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        assert self.trainer.max_epochs != -1
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.trainer.max_epochs
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
