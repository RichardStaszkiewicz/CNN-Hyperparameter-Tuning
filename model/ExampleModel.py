import os
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from filelock import FileLock
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from .mlp import MLP
from .resnet import ResNet

class FashionMNISTClassifier(pl.LightningModule):
    def __init__(self, config):
        super(FashionMNISTClassifier, self).__init__()
        self.accuracy = Accuracy()
        self.mlp = MLP(config["mlp_config"])
        self.resnet = ResNet(config["resnet_config"])
        self.lr = config["lr"]

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, self.layer_1_size)
        self.layer_2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
        self.layer_3 = torch.nn.Linear(self.layer_2_size, 10)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.resnet(x)
        out = self.mlp(out)
        return out

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.data_dir = os.getcwd()
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup(self, stage=None):
        with FileLock(f"{self.data_dir}.lock"):
            mnist = MNIST(
                self.data_dir, train=True, download=True, transform=self.transform
            )
            self.mnist_train, self.mnist_val = random_split(mnist, [55000, 5000])

            self.mnist_test = MNIST(
                self.data_dir, train=False, download=True, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)