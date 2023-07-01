from typing import List, Dict
import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from ldm.models.diffusion.ddpm import LatentDiffusion
from ..representation_transformer import RepresentationTransformer



class AttentionOnLatentDiffusion(pl.LightningModule):
    def __init__(self,
                 trained_diffusion: LatentDiffusion,
                 attention_config: Dict,
                 lr: float=0.001) -> None:
        super().__init__()
        self.trained_diffusion = trained_diffusion
        self.attention_classifier = RepresentationTransformer(**attention_config)
        self.lr = lr


    @torch.no_grad()
    def get_imgs_representation(self, imgs: torch.Tensor):
        encoder_posterior = self.trained_diffusion.encode_first_stage(imgs)
        z = self.trained_diffusion.get_first_stage_encoding(encoder_posterior).detach()
        hs = self.trained_diffusion.model.diffusion_model.just_representations(z, torch.ones(z.shape[0], device=self.device), pooled=False)
        return hs

    def forward(self, imgs: torch.Tensor):
        z = self.get_imgs_representation(imgs)
        return self.attention_classifier(z)

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
