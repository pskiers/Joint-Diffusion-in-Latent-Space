import torch
import torch.nn as nn
from einops import rearrange
import kornia as K
from .joint_latent_diffusion import JointLatentDiffusion
from .representation_transformer import RepresentationTransformer


class SSLJointDiffusion(JointLatentDiffusion):
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
        self.supervised_batch_size = 256
        self.supervised_imgs = None
        self.supervised_labels = None

    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False, cond_key=None, return_original_cond=False, bs=None):
        model_input = super().get_input(batch, k, return_first_stage_outputs, force_c_encode, cond_key, return_original_cond, bs)
        if self.training is True:
            sup_idx = (self.batch_classes >= 0).nonzero(as_tuple=True)
            sup_labels = self.batch_classes[sup_idx]
            sup_imgs = model_input[0][sup_idx]
            if self.supervised_labels is not None:
                self.supervised_labels = torch.cat([self.supervised_labels, sup_labels])
            else:
                self.supervised_labels = sup_labels
            if self.supervised_imgs is not None:
                self.supervised_imgs = torch.cat([self.supervised_imgs, sup_imgs])
            else:
                self.supervised_imgs = sup_imgs
            self.batch_classes = None
            self.x_start = None
        return model_input

    def p_losses(self, x_start, cond, t, noise=None):
        loss, loss_dict = super().p_losses(x_start, cond, t, noise)
        if self.supervised_imgs is not None and len(self.supervised_imgs) > self.supervised_batch_size:
            sup_imgs = self.supervised_imgs[:self.supervised_batch_size]
            sup_labels = self.supervised_labels[:self.supervised_batch_size]

            self.supervised_imgs = self.supervised_imgs[self.supervised_batch_size:]
            self.supervised_labels = self.supervised_labels[self.supervised_batch_size:]

            if isinstance(cond, dict):
                # hybrid case, cond is exptected to be a dict
                pass
            else:
                if not isinstance(cond, list):
                    cond = [cond]
                key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
                cond = {key: cond}

            representations = self.model.diffusion_model.just_representations(
                sup_imgs,
                torch.ones(sup_imgs.shape[0], device=self.device),
                pooled=False
            )
            representations = self.transform_representations(representations)
            preds = self.classifier(representations)

            prefix = 'train' if self.training else 'val'

            loss_classification = nn.functional.cross_entropy(preds, sup_labels)
            loss += loss_classification
            loss_dict.update({f'{prefix}/loss_classification': loss_classification})
            loss_dict.update({f'{prefix}/loss': loss})
            accuracy = torch.sum(torch.argmax(preds, dim=1) == sup_labels) / self.supervised_batch_size
            loss_dict.update({f'{prefix}/accuracy': accuracy})
        return loss, loss_dict


class SSLJointDiffusionV2(JointLatentDiffusion):
    def __init__(
            self,
            first_stage_config,
            cond_stage_config,
            classifier_in_features,
            classifier_hidden,
            num_classes,
            classification_loss_scale=1.0,
            supervised_skip=0,
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
        self.supervised_imgs = None
        self.supervised_labels = None
        self.supervised_dataloader = None
        self.supervised_iterator = None
        self.supervised_skip_n = supervised_skip
        self.supervised_skip_current = supervised_skip
        self.classification_loss_scale = classification_loss_scale

        img_size = first_stage_config['params']['ddconfig']['resolution']
        self.augmentation = K.augmentation.ImageSequential(
            K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.25),
            K.augmentation.RandomResizedCrop((img_size, img_size), scale=(0.5, 1), p=0.25),
            K.augmentation.RandomRotation((-30, 30), p=0.25),
            # K.augmentation.RandomHorizontalFlip(0.5),
            K.augmentation.RandomContrast((0.6, 1.8), p=0.25),
            K.augmentation.RandomSharpness((0.4, 2), p=0.25),
            K.augmentation.RandomBrightness((0.6, 1.8), p=0.25),
            K.augmentation.RandomMixUpV2(p=0.5),
        )
        if kwargs.get("ckpt_path", None) is not None:
            ignore_keys = kwargs.get("ignore_keys", [])
            only_model = kwargs.get("load_only_unet", False)
            self.init_from_ckpt(kwargs["ckpt_path"], ignore_keys=ignore_keys, only_model=only_model)

    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False, cond_key=None, return_original_cond=False, bs=None):
        model_input = super().get_input(batch, k, return_first_stage_outputs, force_c_encode, cond_key, return_original_cond, bs)
        if self.training is True:
            # delete labels for unsupervised data
            self.batch_classes = None
            self.x_start = None

            if self.supervised_skip_current > 0:
                self.supervised_skip_current -= 1
            else:
                try:
                    b = next(self.supervised_iterator)
                except (StopIteration, TypeError):
                    self.supervised_iterator = iter(self.supervised_dataloader)
                    b = next(self.supervised_iterator)
                sup_imgs = b[k]
                if len(sup_imgs.shape) == 3:
                    sup_imgs = sup_imgs[..., None]
                sup_imgs = rearrange(sup_imgs, 'b h w c -> b c h w')
                sup_imgs = sup_imgs.to(memory_format=torch.contiguous_format).float().to(self.device)
                sup_imgs = self.augmentation(sup_imgs)
                sup_imgs = self.encode_first_stage(sup_imgs)
                self.supervised_imgs = self.get_first_stage_encoding(sup_imgs).detach()
                self.supervised_labels = b[self.classification_key].to(self.device)
                self.supervised_skip_current = self.supervised_skip_n

        return model_input

    def p_losses(self, x_start, cond, t, noise=None):
        loss, loss_dict = super().p_losses(x_start, cond, t, noise)
        if self.supervised_imgs is not None:
            if isinstance(cond, dict):
                # hybrid case, cond is exptected to be a dict
                pass
            else:
                if not isinstance(cond, list):
                    cond = [cond]
                key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
                cond = {key: cond}

            representations = self.model.diffusion_model.just_representations(
                self.supervised_imgs,
                torch.ones(self.supervised_imgs.shape[0], device=self.device),
                pooled=False
            )
            representations = self.transform_representations(representations)
            preds = self.classifier(representations)

            prefix = 'train' if self.training else 'val'

            loss_classification = nn.functional.cross_entropy(preds, self.supervised_labels) * self.classification_loss_scale
            loss += loss_classification
            loss_dict.update(
                {f'{prefix}/loss_classification': loss_classification})
            loss_dict.update({f'{prefix}/loss': loss})
            accuracy = torch.sum(torch.argmax(preds, dim=1) == self.supervised_labels) / len(self.supervised_labels)
            loss_dict.update({f'{prefix}/accuracy': accuracy})

            self.supervised_imgs = None
            self.supervised_labels = None
        return loss, loss_dict


class SSLJointDiffusionV3(SSLJointDiffusionV2):
    def __init__(
            self,
            first_stage_config,
            cond_stage_config,
            attention_config,
            classification_loss_scale=1.0,
            supervised_skip=0,
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
            classification_loss_scale,
            supervised_skip,
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
