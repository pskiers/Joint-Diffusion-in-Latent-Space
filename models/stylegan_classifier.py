import torch
import torch.nn as nn
from typing import List


class ClassifierHead(nn.Module):
    def __init__(self, channels: int, projection_div: int = 1, out_size: int = 1) -> None:
        super().__init__()
        self.inner_dim = channels // projection_div
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
        self.conv_out = nn.Conv2d(in_channels=self.inner_dim, out_channels=out_size, kernel_size=1)

    def forward(self, x_B_C_W_H: torch.Tensor) -> torch.Tensor:
        residual = self.proj_in(x_B_C_W_H)
        out = self.conv_res(residual) + residual
        return self.conv_out(out).mean(dim=3).mean(dim=2)


class StyleGANDiscriminator(nn.Module):
    def __init__(self, context_dims: List[int], projection_div: int = 1) -> None:
        super().__init__()
        self.context_dims = context_dims
        self.heads = nn.ModuleList(
            [ClassifierHead(channels=dim, projection_div=projection_div) for dim in context_dims]
        )

    def forward(self, representations: List[torch.Tensor]) -> torch.Tensor:
        logits = [head(repr) for repr, head in zip(representations, self.heads)]
        return torch.cat(logits, dim=1)
