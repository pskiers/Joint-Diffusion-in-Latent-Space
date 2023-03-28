import torch
import torch.nn as nn
import pytorch_lightning as pl
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.diffusionmodules.util import timestep_embedding



class ClassifierOnLatentDiffusion(pl.LightningModule):
    def __init__(
            self, trained_diffusion: LatentDiffusion,
            num_classes: int,
            lr: float=0.001
        ) -> None:
        super().__init__()
        self.trained_diffusion = trained_diffusion
        self.num_classes = num_classes
        self.avg_pool = nn.AvgPool2d(2)
        self.mlp = nn.Sequential(
            nn.Linear(8064, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.num_classes)
        )
        self.lr = lr

    @torch.no_grad()
    def get_imgs_representation(self, imgs: torch.Tensor):
        encoder_posterior = self.trained_diffusion.encode_first_stage(imgs)
        z = self.trained_diffusion.get_first_stage_encoding(encoder_posterior).detach()
        hs = []
        t_emb = timestep_embedding(torch.ones(z.shape[0], device=self.device), self.trained_diffusion.model.diffusion_model.model_channels, repeat_only=False)
        emb = self.trained_diffusion.model.diffusion_model.time_embed(t_emb)

        for module in self.trained_diffusion.model.diffusion_model.input_blocks:
            z = module(z, emb, None)
            hs.append(z)
        return hs

    def forward(self, imgs: torch.Tensor):
        z = self.get_imgs_representation(imgs)
        z = [self.avg_pool(z_i) for z_i in z]
        z = [torch.flatten(z_i, start_dim=1) for z_i in z]
        z = torch.concat(z, dim=1)
        return self.mlp(z)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch[0]
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
