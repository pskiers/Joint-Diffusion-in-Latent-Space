import torch
import torch.nn as nn
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.diffusionmodules.util import timestep_embedding



class ClassifierOnLatentDiffusion(nn.Module):
    def __init__(self, trained_diffusion: LatentDiffusion, num_classes: int, device: torch.device) -> None:
        super().__init__()
        self.trained_diffusion = trained_diffusion
        self.num_classes = num_classes
        self.device = device
        self.avg_pool = nn.AvgPool2d(2)
        self.mlp = nn.Sequential(
            nn.Linear(8064, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.num_classes)
        )
        self.to(device)

    @torch.no_grad()
    def get_imgs_representation(self, imgs: torch.Tensor):
        encoder_posterior = self.trained_diffusion.encode_first_stage(imgs.to(self.device))
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
