import torch
from .joint_latent_diffusion_noisy_classifier import JointLatentDiffusionNoisyClassifier, JointLatentDiffusionNoisyAttention
from ..representation_transformer import RepresentationTransformer


class JointLatentDiffusion(JointLatentDiffusionNoisyClassifier):
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 classifier_in_features,
                 classifier_hidden,
                 num_classes,
                 dropout=0,
                 classification_loss_weight=1.0,
                 classification_key=1,
                 augmentations=True,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1,
                 scale_by_std=False,
                 *args,
                 **kwargs):
        super().__init__(
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            classifier_in_features=classifier_in_features,
            classifier_hidden=classifier_hidden,
            num_classes=num_classes,
            dropout=dropout,
            classification_loss_weight=classification_loss_weight,
            sample_grad_scale=0,
            classification_key=classification_key,
            augmentations=augmentations,
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
        # self.x_start = None
        self.gradient_guided_sampling = False

    def train_classification_step(self, batch, loss):
        loss_dict = {}

        x, y = self.get_train_classification_input(batch, self.first_stage_key)
        t = torch.zeros((x.shape[0],), device=self.device).long()

        loss_classification, accuracy = self.do_classification(x, t, y)
        loss += loss_classification * self.classification_loss_weight

        loss_dict.update({'train/loss_classification': loss_classification})
        loss_dict.update({'train/loss_full': loss})
        loss_dict.update({'train/accuracy': accuracy})

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = self.get_valid_classification_input(batch, self.first_stage_key)
        t = torch.zeros((x.shape[0],), device=self.device).long()
        x_diff, c = self.get_input(batch, self.first_stage_key)

        loss, loss_dict_no_ema = self(x_diff, c)
        loss_cls, accuracy = self.do_classification(x, t, y)
        loss_dict_no_ema.update({'val/loss_classification': loss_cls})
        loss_dict_no_ema.update({'val/loss_full': loss + loss_cls})
        loss_dict_no_ema.update({'val/accuracy': accuracy})

        with self.ema_scope():
            loss, loss_dict_ema = self(x_diff, c)
            loss_cls, accuracy = self.do_classification(x, t, y)
            loss_dict_ema.update({'val/loss_classification': loss_cls})
            loss_dict_ema.update({'val/loss_full': loss + loss_cls})
            loss_dict_ema.update({'val/accuracy': accuracy})
            loss_dict_ema = {
                key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}

        self.log_dict(
            loss_dict_no_ema,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )
        self.log_dict(
            loss_dict_ema,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )


class JointLatentDiffusionAttention(JointLatentDiffusionNoisyAttention):
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 attention_config,
                 augmentations=True,
                 classification_loss_weight=1.0,
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
                 **kwargs):
        super().__init__(
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            attention_config=attention_config,
            augmentations=augmentations,
            classification_loss_weight=classification_loss_weight,
            sample_grad_scale=0,
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
        self.gradient_guided_sampling = False

    def train_classification_step(self, batch, loss):
        loss_dict = {}

        x, y = self.get_train_classification_input(batch, self.first_stage_key)
        t = torch.zeros((x.shape[0],), device=self.device).long()

        loss_classification, accuracy = self.do_classification(x, t, y)
        loss += loss_classification * self.classification_loss_weight

        loss_dict.update({'train/loss_classification': loss_classification})
        loss_dict.update({'train/loss_full': loss})
        loss_dict.update({'train/accuracy': accuracy})

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = self.get_valid_classification_input(batch, self.first_stage_key)
        t = torch.zeros((x.shape[0],), device=self.device).long()

        loss, loss_dict_no_ema = self.shared_step(batch)
        loss_cls, accuracy = self.do_classification(x, t, y)
        loss_dict_no_ema.update({'val/loss_classification': loss_cls})
        loss_dict_no_ema.update({'val/loss_full': loss + loss_cls})
        loss_dict_no_ema.update({'val/accuracy': accuracy})

        with self.ema_scope():
            loss, loss_dict_ema = self.shared_step(batch)
            loss_cls, accuracy = self.do_classification(x, t, y)
            loss_dict_ema.update({'val/loss_classification': loss_cls})
            loss_dict_ema.update({'val/loss_full': loss + loss_cls})
            loss_dict_ema.update({'val/accuracy': accuracy})
            loss_dict_ema = {
                key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}

        self.log_dict(
            loss_dict_no_ema,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )
        self.log_dict(
            loss_dict_ema,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )
