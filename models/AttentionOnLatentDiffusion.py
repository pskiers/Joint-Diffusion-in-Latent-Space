from typing import List
import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.diffusionmodules.openaimodel import SpatialTransformer
from ldm.modules.attention import Normalize



class AttentionOnLatentDiffusion(pl.LightningModule):
    def __init__(
            self,
            trained_diffusion: LatentDiffusion,
            num_classes: int,
            channels: int,
            dim_head: int,
            context_dims: List[int],
            mlp_size: int,
            hidden_size: int,
            lr: float=0.001
        ) -> None:
        super().__init__()
        self.trained_diffusion = trained_diffusion
        self.num_classes = num_classes
        self.lr = lr
        self.attention_blocks = nn.ModuleList([
            SpatialTransformer(in_channels=channels, n_heads=channels//dim_head, d_head=dim_head, context_dim=context_dim)
            for context_dim in context_dims
        ])
        self.context_projections = nn.ModuleList([
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
            for channel in context_dims
        ])
        self.norms = nn.ModuleList([
            Normalize(channel)
            for channel in context_dims
        ])
        self.mlp = nn.Sequential(
            nn.Linear(mlp_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_size, num_classes)
        )

    @torch.no_grad()
    def get_imgs_representation(self, imgs: torch.Tensor):
        encoder_posterior = self.trained_diffusion.encode_first_stage(imgs)
        z = self.trained_diffusion.get_first_stage_encoding(encoder_posterior).detach()
        _, hs = self.trained_diffusion.model(z, torch.ones(z.shape[0], device=self.device))
        return hs

    def forward(self, imgs: torch.Tensor):
        z = self.get_imgs_representation(imgs)
        x = None
        for z_i, transformer, projection, norm in zip(z, self.attention_blocks, self.context_projections, self.norms):
            context = norm(z_i)
            context = projection(context)
            context = rearrange(context, 'b c h w -> b (h w) c')
            if x is None:
                x = transformer(z_i, context=context)
            else:
                x = transformer(x, context=context)
        return self.mlp(x.flatten(start_dim=1))

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
