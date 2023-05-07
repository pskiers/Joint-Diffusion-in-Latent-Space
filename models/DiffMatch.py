import torch
import torch.nn as nn
from einops import rearrange
from .SSLJointDiffusion import SSLJointDiffusion


class DiffMatch(SSLJointDiffusion):
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
            first_stage_config,
            cond_stage_config,
            classifier_in_features,
            classifier_hidden,
            num_classes,
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
        self.min_confidence = 0.7
        self.raw_imgs = None

    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False, cond_key=None, return_original_cond=False, bs=None):
        self.raw_imgs = batch[k]
        if len(self.raw_imgs.shape) == 3:
            self.raw_imgs = self.raw_imgs[..., None]
        self.raw_imgs = rearrange(self.raw_imgs, 'b h w c -> b c h w')
        self.raw_imgs = self.raw_imgs.to(memory_format=torch.contiguous_format).float()
        if bs is not None:
            self.raw_imgs = self.raw_imgs[:bs]
        self.raw_imgs = self.raw_imgs.to(self.device)

        return super().get_input(batch, k, return_first_stage_outputs, force_c_encode, cond_key, return_original_cond, bs)

    def p_losses(self, x_start, cond, t, noise=None):
        loss, loss_dict = super().p_losses(x_start, cond, t, noise)

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        weakly_augmented = self.weakly_augment(self.raw_imgs)
        encoder_posterior_weak = self.encode_first_stage(weakly_augmented)
        z_weak = self.get_first_stage_encoding(encoder_posterior_weak).detach()

        _, weak_rep = self.model(z_weak, torch.ones(z_weak.shape[0], device=self.device), **cond)
        weak_rep = [torch.flatten(z_i, start_dim=1) for z_i in weak_rep]
        weak_rep = torch.concat(weak_rep, dim=1)
        weak_preds = nn.functional.softmax(self.classifier(weak_rep), dim=1)
        pseudo_labels = weak_preds.argmax(dim=1)
        above_threshold_idx = (weak_preds.max(dim=1).values > self.min_confidence).nonzero(as_tuple=True)
        pseudo_labels = pseudo_labels[above_threshold_idx]

        strongly_augmented = self.strongly_augment(self.raw_imgs[above_threshold_idx])
        encoder_posterior_strong = self.encode_first_stage(strongly_augmented)
        z_strong = self.get_first_stage_encoding(encoder_posterior_strong).detach()

        _, strong_rep = self.model(z_strong, torch.ones(z_strong.shape[0], device=self.device), **cond)
        strong_rep = [torch.flatten(z_i, start_dim=1) for z_i in strong_rep]
        strong_rep = torch.concat(strong_rep, dim=1)
        preds = self.classifier(strong_rep)
        ssl_loss = nn.functional.cross_entropy(preds, pseudo_labels)

        prefix = 'train' if self.training else 'val'
        loss += ssl_loss
        loss_dict.update({f'{prefix}/loss_ssl_classification': ssl_loss})
        loss_dict.update({f'{prefix}/loss': loss})
        accuracy = torch.sum(torch.argmax(preds, dim=1) == pseudo_labels) / len(pseudo_labels)
        loss_dict.update({f'{prefix}/ssl_accuracy': accuracy})

        return loss, loss_dict

    def weakly_augment(self, imgs):
        # TODO
        pass

    def strongly_augment(self, imgs):
        # TODO
        pass