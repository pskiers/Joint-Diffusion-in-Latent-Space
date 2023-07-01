from typing import Any
import torch
import torch.nn as nn
import torchvision as tv
import pytorch_lightning as pl
import kornia as K
from ldm.models.autoencoder import AutoencoderKL


class WideResNet(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.conv = tv.models.resnet18()
        self.relu = nn.ReLU()
        self.head = nn.Linear(1000, 10)
        self.lr = 0.001
        self.augmentation = K.augmentation.ImageSequential(
            K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.25),
            K.augmentation.RandomResizedCrop((32, 32), scale=(0.5, 1), p=0.25),
            K.augmentation.RandomRotation((-30, 30), p=0.25),
            # K.augmentation.RandomHorizontalFlip(0.5),
            K.augmentation.RandomContrast((0.6, 1.8), p=0.25),
            K.augmentation.RandomSharpness((0.4, 2), p=0.25),
            K.augmentation.RandomBrightness((0.6, 1.8), p=0.25),
            K.augmentation.RandomMixUpV2(p=0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)
        return self.head(out)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs = imgs.transpose(1, 3)
        imgs = self.augmentation(imgs)
        preds = self(imgs)

        loss = nn.functional.cross_entropy(preds, labels)
        acc = torch.sum(torch.argmax(preds, dim=1) == labels) / len(labels)
        loss_dict = {"train/loss": loss, "train/accuracy": acc}
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs = imgs.transpose(1, 3)
        preds = self(imgs)

        loss = nn.functional.cross_entropy(preds, labels)
        acc = torch.sum(torch.argmax(preds, dim=1) == labels) / len(labels)
        loss_dict = {"val/loss": loss, "val/accuracy": acc}
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=False, on_epoch=False)
        return loss_dict

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class WideResNetEncoder(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.projection = nn.Conv2d(4, 3, 1)
        self.conv = tv.models.wide_resnet50_2()
        self.relu = nn.ReLU()
        self.head = nn.Linear(1000, 10)
        self.lr = 0.001
        self.augmentation = K.augmentation.ImageSequential(
            K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.25),
            K.augmentation.RandomResizedCrop((32, 32), scale=(0.5, 1), p=0.25),
            K.augmentation.RandomRotation((-30, 30), p=0.25),
            # K.augmentation.RandomHorizontalFlip(0.5),
            K.augmentation.RandomContrast((0.6, 1.8), p=0.25),
            K.augmentation.RandomSharpness((0.4, 2), p=0.25),
            K.augmentation.RandomBrightness((0.6, 1.8), p=0.25),
            K.augmentation.RandomMixUpV2(p=0.5),
        )
        self.encoder = AutoencoderKL(
            embed_dim=4,
            monitor="val/rec_loss",
            ckpt_path="logs/Autoencoder_2023-04-15T23-50-00/checkpoints/last.ckpt",
            ddconfig={
              "double_z": True,
              "z_channels": 4,
              "resolution": 32,
              "in_channels": 3,
              "out_ch": 3,
              "ch": 128,
              "ch_mult": [1, 2, 4],  # num_down = len(ch_mult)-1
              "num_res_blocks": 1,
              "attn_resolutions": [],
              "dropout": 0.0,
            },
            lossconfig={
                "target": "torch.nn.Identity"
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.projection(x)
        out = self.conv(out)
        out = self.relu(out)
        return self.head(out)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs = imgs.transpose(1, 3)
        imgs = self.augmentation(imgs)
        imgs = self.encoder.encode(imgs).sample()

        preds = self(imgs)

        loss = nn.functional.cross_entropy(preds, labels)
        acc = torch.sum(torch.argmax(preds, dim=1) == labels) / len(labels)
        loss_dict = {"train/loss": loss, "train/accuracy": acc}
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs = imgs.transpose(1, 3)
        imgs = self.encoder.encode(imgs).sample()
        preds = self(imgs)

        loss = nn.functional.cross_entropy(preds, labels)
        acc = torch.sum(torch.argmax(preds, dim=1) == labels) / len(labels)
        loss_dict = {"val/loss": loss, "val/accuracy": acc}
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=False, on_epoch=False)
        return loss_dict

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

