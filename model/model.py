from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import pytorch_lightning as pl
import numpy as np
from .resnet import ResNet
from .mlp import MLP
from pytorch_metric_learning import distances, miners, losses
import torch.optim as optim
from torchmetrics import Accuracy


class ExampleModel(pl.LightningModule):

    def __init__(self,
                 regularization_ratio: float,
                 resnet_config: dict,
                 mlp_config: dict,
                 lr: float,
                 img_key: Any = 0,
                 class_key: Any = 1,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.resnet = ResNet(**resnet_config)
        self.mlp = MLP(**mlp_config)
        self.regularization_ratio = regularization_ratio
        self.img_key = img_key
        self.bh_kwargs = bh_kwargs if bh_kwargs is not None else {}
        self.class_key = class_key


        self.val_emb = None
        self.val_labels = None
        self.test_emb = None
        self.test_labels = None
        self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.backbone(x.float())
        loss = self.criterion(y_pred, y)

        accuracy = Accuracy("multiclass", num_classes=10).to("cuda")
        acc = accuracy(y_pred, y)

        self.log("train/accuracy", acc, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.backbone(x.float())
        loss = self.criterion(y_pred, y)

        accuracy = Accuracy("multiclass", num_classes=10).to("cuda")
        acc = accuracy(y_pred, y)

        self.log("val/accuracy", acc, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.backbone.parameters(), lr=1e-3)
        return optimizer

    def forward(self, imgs: torch.Tensor) -> tuple:
        """
        return -> tuple[torch.Tensor, torch.Tensor]
        """
        out = self.resnet(imgs)
        out = self.mlp(out)
        latent_dim = out.shape[1] // 2
        mean, std = out[:, :latent_dim], out[:, latent_dim:]
        return mean, std

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        imgs = batch[self.img_key]
        labels = batch[self.class_key]
        means, stds = self(imgs)
        embeds = self.get_embeds(means, stds)
        loss = self.batch_handler(
            embeds, labels, self.exp_class_distance, self.regularization_ratio, **self.bh_kwargs)

        self.test_emb = torch.cat(
            [self.test_emb, embeds], dim=0) if self.test_emb is not None else embeds
        self.test_labels = torch.cat(
            [self.test_labels, labels], dim=0) if self.test_labels is not None else labels

        loss_dict = {"test/loss": loss}
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        if isinstance(self.test_emb, torch.Tensor):
            hidden = self.test_emb.shape[1] // 2
            self.test_emb = (self.test_emb[:, :hidden], self.test_emb[:, hidden:])

        if self.precision_k is None:
            self.precision_k = [0, 2, 4]
        if self.recall_k is None:
            self.recall_k = [0, 2, 4]

        prec_dict = precision_at_k(self.test_emb, self.test_labels, self.precision_k)
        log_dict = {f"test/precision@{k}": prec_dict[k] for k in prec_dict.keys()}
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        recall_dict = recall_at_k(self.test_emb, self.test_labels, self.recall_k)
        log_dict = {f"test/recall@{k}": recall_dict[k] for k in recall_dict.keys()}
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.test_emb = None
        self.test_labels = None

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optim