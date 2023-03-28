from ldm.models.diffusion.ddpm import LatentDiffusion


class JointDiffusionInLatentSpace(LatentDiffusion):
    def __init__(
            self,
            first_stage_config,
            cond_stage_config,
            num_timesteps_cond=None,
            cond_stage_key="image",
            cond_stage_trainable=False,
            concat_mode=True,
            cond_stage_forward=None,
            conditioning_key=None,
            scale_factor=1,
            scale_by_std=False,
            *args,
            **kwargs
        ):
        super().__init__(
            first_stage_config,
            cond_stage_config,
            num_timesteps_cond,
            cond_stage_key,
            cond_stage_trainable,
            concat_mode,
            cond_stage_forward,
            conditioning_key,
            scale_factor,
            scale_by_std,
            *args,
            **kwargs
        )

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        return super().apply_model(x_noisy, t, cond, return_ids)

    def p_losses(self, x_start, cond, t, noise=None):
        return super().p_losses(x_start, cond, t, noise)

