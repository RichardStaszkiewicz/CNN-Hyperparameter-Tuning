import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets

from model.mlp import MLP
from model.resnet import ResNet
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train_transform=None,
        test_transform=None,
        image_size=None,
        train_valid_split=None,
    ):
        super().__init__()
        self.image_size = image_size if image_size is not None else 28
        self.train_transform = (
            train_transform
            if train_transform is not None
            else transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(30),
                    transforms.Resize(self.image_size),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                ]
            )
        )
        self.test_transform = (
            test_transform
            if test_transform is not None
            else transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                ]
            )
        )

        self.batch_size = batch_size  # config["batch_size"]
        self.train_valid_split = (
            train_valid_split if train_valid_split is not None else 0.8
        )

    def setup(self, stage=None):
        whole_train_dataset = datasets.FashionMNIST(
            root="data", train=True, transform=self.train_transform, download=True
        )
        train_size = int(self.train_valid_split * len(whole_train_dataset))
        valid_size = len(whole_train_dataset) - train_size
        self.mnist_train, self.mnist_val = random_split(
            whole_train_dataset, [train_size, valid_size]
        )
        self.mnist_test = datasets.FashionMNIST(
            root="data", train=False, transform=self.test_transform, download=True
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=2)


class MNISTClassifier(pl.LightningModule):
    def __init__(self, config):
        super(MNISTClassifier, self).__init__()
        self.accuracy = Accuracy("multiclass", num_classes=10)
        self.mlp = MLP(**config["mlp_config"])
        self.resnet = ResNet(**config["resnet_config"])
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.betas = (config["beta_1"], config["beta_2"])

        self.validation_step_outputs = []

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def forward(self, x):
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
        self.validation_step_outputs.append(
            {"val_loss": loss, "val_accuracy": accuracy}
        )

        return {"val_loss": loss, "val_accuracy": accuracy}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(
            [x["val_loss"] for x in self.validation_step_outputs]
        ).mean()
        avg_acc = torch.stack(
            [x["val_accuracy"] for x in self.validation_step_outputs]
        ).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        return optimizer
