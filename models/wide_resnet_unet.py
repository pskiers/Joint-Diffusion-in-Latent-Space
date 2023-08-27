from typing import List
import torch
import torch.nn as nn
from ldm.modules.diffusionmodules.openaimodel import (
    ResBlock,
    AttentionBlock,
    TimestepEmbedSequential,
    Upsample,
    normalization,
    zero_module,
    conv_nd,
    timestep_embedding,
    linear
)
from .wide_resnet import Wide_ResNet, Wide_ResNet_Timestep


class Wide_ResNet_UNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 out_channels: int,
                 repr_channels: List[int],
                 decoder_channels_mult: List[int],
                 ds: int,  # downsample done by
                 model_channels: int,
                 attention_resolutions: List[int],
                 num_res_blocks: int = 2,
                 dims: int = 2,
                 num_heads=-1,
                 num_head_channels=-1,
                 num_heads_upsample=-1,
                 use_checkpoint=False,
                 resblock_updown=False,
                 conv_resample=True,
                 use_scale_shift_norm=False,
                 use_new_attention_order=False,
                 legacy=True,
                 depth: int = 28,
                 widen_factor: int = 2,
                 dropout: float = 0.0,
                 unet_layer_idx: List[int] = [0, 1, 2, 3],
                 use_timestep_emb=True) -> None:
        super().__init__()
        self.model_channels = model_channels
        self.representations = []
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        def hook(module, input, output):
            self.representations.append(output)
        self.use_timestep_emb = use_timestep_emb
        if use_timestep_emb:
            self.encoder = Wide_ResNet_Timestep(depth, widen_factor, dropout, num_classes, time_embed_dim)
        else:
            self.encoder = Wide_ResNet(depth, widen_factor, dropout, num_classes)
        target_layers = []
        for layer in [self.encoder.block1,
                      self.encoder.block2,
                      self.encoder.block3]:
            target_layers.extend([layer.layer[idx] for idx in unet_layer_idx])
        for layer in target_layers:
            layer.register_forward_hook(hook)

        time_embed_dim = model_channels * 4
        ch = repr_channels.pop()
        self.output_blocks = nn.ModuleList([])
        for level, mult in enumerate(decoder_channels_mult):
            for i in range(num_res_blocks + 1):
                ich = repr_channels.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level != len(decoder_channels_mult) - 1 and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(
                conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                pooled=True,
                **kwargs):
        emb = self.get_timestep_embedding(x, timesteps, y)

        if self.use_timestep_emb:
            out = self.encoder(x, emb)
        else:
            out = self.encoder(x)

        rec = self.forward_output_blocks(x, context, emb, self.representations)
        self.representations = []
        return rec, out

    def just_representations(self,
                             x,
                             timesteps=None,
                             context=None,
                             y=None,
                             pooled=True,
                             **kwargs):
        if self.use_timestep_emb:
            emb = self.get_timestep_embedding(x, timesteps, y)
            out = self.encoder(x, emb)
        else:
            out = self.encoder(x)
        self.representations = []
        return out

    def just_reconstruction(self,
                            x,
                            timesteps=None,
                            context=None,
                            y=None,
                            **kwargs):
        emb = self.get_timestep_embedding(x, timesteps, y)

        if self.use_timestep_emb:
            self.encoder(x, emb)
        else:
            self.encoder(x)

        rec = self.forward_output_blocks(x, context, emb, self.representations)
        self.representations = []
        return rec

    def forward_output_blocks(self, x, context, emb, representations):
        h = representations[-1]
        for module in self.output_blocks:
            last_hs = representations.pop()
            h = torch.cat([h, last_hs], dim=1)
            h = module(h, emb, context)
            representations.insert(0, last_hs)
        h = h.type(x.dtype)

        return self.out(h)

    def get_timestep_embedding(self, x, timesteps, y):
        t_emb = timestep_embedding(
            timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        return emb
