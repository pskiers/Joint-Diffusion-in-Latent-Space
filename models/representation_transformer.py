from typing import List
import torch
import torch.nn as nn
from einops import rearrange
from ldm.modules.diffusionmodules.openaimodel import SpatialTransformer
from ldm.modules.attention import Normalize


class RepresentationTransformer(nn.Module):
    def __init__(
            self,
            num_classes: int,
            channels: int,
            dim_head: int,
            context_dims: List[int],
            mlp_size: int,
            hidden_size: int,
        ) -> None:
        super().__init__()
        self.attn_blocks = nn.ModuleList([
            SpatialTransformer(in_channels=channels,
                               n_heads=channels//dim_head,
                               d_head=dim_head,
                               context_dim=context_dim)
            for context_dim in context_dims
        ])
        self.repr_projections = nn.ModuleList([
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

    def forward(self, repr: List[torch.Tensor]) -> torch.Tensor:
        x = None
        for z_i, transformer, projection, norm in zip(repr, self.attn_blocks, self.repr_projections, self.norms):
            context = norm(z_i)
            context = projection(context)
            context = rearrange(context, 'b c h w -> b (h w) c')
            if x is None:
                x = transformer(z_i, context=context)
            else:
                x = transformer(x, context=context)
        return self.mlp(x.flatten(start_dim=1))
