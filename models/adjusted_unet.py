import torch as th
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.diffusionmodules.util import timestep_embedding


class AdjustedUNet(UNetModel):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=...,
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,
            transformer_depth=1,
            context_dim=None,
            n_embed=None,
            legacy=True,
            pool_size=2
        ):
        super().__init__(
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout,
            channel_mult,
            conv_resample,
            dims,
            num_classes,
            use_checkpoint,
            use_fp16,
            num_heads,
            num_head_channels,
            num_heads_upsample,
            use_scale_shift_norm,
            resblock_updown,
            use_new_attention_order,
            use_spatial_transformer,
            transformer_depth,
            context_dim,
            n_embed,
            legacy
        )
        self.pool_size = pool_size

    def forward(self, x, timesteps=None, context=None, y=None, pooled=True, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs and pooled representations.
        """
        emb = self.get_timestep_embedding(x, timesteps, y)

        representations = self.forward_input_blocks(x, context, emb)

        output = self.forward_output_blocks(x, context, emb, representations)

        if pooled is True:
            representations = self.pool_representations(representations)

        return output, representations

    def just_representations(self, x, timesteps=None, context=None, y=None, pooled=True, **kwargs):
        emb = self.get_timestep_embedding(x, timesteps, y)

        representations = self.forward_input_blocks(x, context, emb)

        if pooled is True:
            representations = self.pool_representations(representations)

        return representations

    def just_reconstruction(self, x, timesteps=None, context=None, y=None, **kwargs):
        emb = self.get_timestep_embedding(x, timesteps, y)

        representations = self.forward_input_blocks(x, context, emb)

        output = self.forward_output_blocks(x, context, emb, representations)

        return output

    def pool_representations(self, representations):
        pooled_representations = []
        for h in representations:
            if h.shape[-1] < self.pool_size:
                pooled_representations.append(th.nn.functional.avg_pool2d(h, h.shape[-1]))
            else:
                pooled_representations.append(th.nn.functional.avg_pool2d(h, self.pool_size))
        return pooled_representations

    def forward_output_blocks(self, x, context, emb, representations):
        h = representations.pop()
        representations.insert(0, h)
        for module in self.output_blocks:
            last_hs = representations.pop()
            h = th.cat([h, last_hs], dim=1)
            h = module(h, emb, context)
            representations.insert(0, last_hs)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)

    def forward_input_blocks(self, x, context, emb):
        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        hs.append(h)
        return hs

    def get_timestep_embedding(self, x, timesteps, y):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(dtype=th.bfloat16)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        return emb
