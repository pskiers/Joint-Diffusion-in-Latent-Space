import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional


class ClassifierHead(nn.Module):
    def __init__(self, channels: int, projection_div: int = 1, out_size: int = 1, emb_dim: Optional[int] = None, emb_proj: Optional[int] = None) -> None:
        super().__init__()
        self.inner_dim = channels // projection_div
        self.emb_proj_dim = emb_proj
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=self.inner_dim, kernel_size=1),
            nn.BatchNorm2d(self.inner_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels=self.inner_dim, out_channels=self.inner_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.inner_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        if emb_proj is None:
            self.conv_out = nn.Conv2d(in_channels=self.inner_dim, out_channels=out_size, kernel_size=1)
        else:
            assert emb_dim is not None
            self.conv_out = nn.Conv2d(in_channels=self.inner_dim, out_channels=emb_proj, kernel_size=1)
            self.emb_proj = nn.Linear(emb_dim, emb_proj)

    def forward(self, x_B_C_W_H: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = self.proj_in(x_B_C_W_H)
        out = self.conv_res(residual) + residual
        out = self.conv_out(out).mean(dim=3).mean(dim=2)
        if emb is not None:
            emb = self.emb_proj(emb)
            assert self.emb_proj_dim is not None
            out = (out * emb).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.emb_proj_dim))
        return out


class StyleGANDiscriminator(nn.Module):
    def __init__(self, context_dims: List[int], projection_div: int = 1, emb_dim: Optional[int] = None, emb_proj: Optional[int] = None) -> None:
        super().__init__()
        self.context_dims = context_dims
        self.heads = nn.ModuleList(
            [ClassifierHead(channels=dim, projection_div=projection_div, emb_dim=emb_dim, emb_proj=emb_proj) for dim in context_dims]
        )

    def forward(self, representations: List[torch.Tensor], emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = [head(repr, emb=emb) for repr, head in zip(representations, self.heads)]
        return torch.cat(logits, dim=1)
