import torch
import torch.nn as nn
import pytorch_lightning as pl
from ..adjusted_unet import AdjustedUNet
from .fixmatch import FixMatch, FixMatchEma


class DiffusionClassifier(pl.LightningModule):
    def __init__(
            self,
            unet_config: dict,
            num_classes: int,
            in_features: int,
            hidden_layer: int,
            lr: float = 0.001
        ) -> None:
        super().__init__()
        self.unet = AdjustedUNet(**unet_config)
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, self.num_classes)
        )
        self.lr = lr

    def forward(self, imgs: torch.Tensor):
        _, z = self.unet(imgs, torch.ones(imgs.shape[0], device=self.device))
        z = [torch.flatten(z_i, start_dim=1) for z_i in z]
        z = torch.concat(z, dim=1)
        return self.mlp(z)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs = imgs.transpose(1, 3)
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


class FixMatchDiffusionClassifier(FixMatch):
    def __init__(self, unet_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = DiffusionClassifier(unet_config, num_classes=10, in_features=3712, hidden_layer=1024)
        self.model_ema = FixMatchEma(self.model, decay=0.999)
