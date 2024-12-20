import torch
import torch.nn as nn
from ldm.modules.diffusionmodules.util import timestep_embedding


class TwoLayerNet(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_layer = nn.Sequential(
            nn.Linear(in_features=in_size, out_features=hidden_size),
            nn.ReLU(),
        )
        time_embed_dim = hidden_size
        self.hidden = hidden_size
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.out_layer = nn.Linear(in_features=hidden_size+time_embed_dim, out_features=out_size)
    
    def get_timestep_embedding(self, timesteps):
        t_emb = timestep_embedding(timesteps, self.hidden, repeat_only=False)
        emb = self.time_embed(t_emb)
        return emb

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        emb = self.get_timestep_embedding(t)
        out = self.in_layer(x)
        out = self.out_layer(torch.cat((out, emb), dim=-1))
        return out
