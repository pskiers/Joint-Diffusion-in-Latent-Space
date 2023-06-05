import torch
from .joint_latent_diffusion_noisy_classifier import JointLatentDiffusionNoisyClassifier
from .representation_transformer import RepresentationTransformer


class JointLatentDiffusion(JointLatentDiffusionNoisyClassifier):
    def __init__(
            self,
            first_stage_config,
            cond_stage_config,
            classifier_in_features,
            classifier_hidden,
            num_classes,
            classification_key=1,
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
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            classifier_in_features=classifier_in_features,
            classifier_hidden=classifier_hidden,
            num_classes=num_classes,
            classification_key=classification_key,
            num_timesteps_cond=num_timesteps_cond,
            cond_stage_key=cond_stage_key,
            cond_stage_trainable=cond_stage_trainable,
            concat_mode=concat_mode,
            cond_stage_forward=cond_stage_forward,
            conditioning_key=conditioning_key,
            scale_factor=scale_factor,
            scale_by_std=scale_by_std,
            *args,
            **kwargs
        )
        self.x_start = None
        self.gradient_guided_sampling = False

    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False, cond_key=None, return_original_cond=False, bs=None):
        out = super().get_input(batch, k, return_first_stage_outputs, force_c_encode, cond_key, return_original_cond, bs)
        self.x_start = out[0]
        return out

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if hasattr(self, "split_input_params"):
            raise NotImplementedError("This feature is not available for this model")

        x_recon = self.model.diffusion_model.just_reconstruction(x_noisy, t)
        if self.x_start is not None:
            representations = self.model.diffusion_model.just_representations(
                self.x_start,
                torch.ones(self.x_start.shape[0], device=self.device),
                pooled=False
            )
            representations = self.transform_representations(representations)
            self.batch_class_predictions = self.classifier(representations)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon


class JointLatentDiffusionAttention(JointLatentDiffusion):
    def __init__(
            self,
            first_stage_config,
            cond_stage_config,
            attention_config,
            classification_key=1,
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
            0,
            0,
            0,
            classification_key,
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
        self.classifier = RepresentationTransformer(**attention_config)
        if kwargs.get("ckpt_path", None) is not None:
            ignore_keys = kwargs.get("ignore_keys", [])
            only_model = kwargs.get("load_only_unet", False)
            self.init_from_ckpt(kwargs["ckpt_path"], ignore_keys=ignore_keys, only_model=only_model)

    def transform_representations(self, representations):
        return representations
