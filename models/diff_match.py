import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange
import kornia as K
from .ssl_joint_diffusion import SSLJointDiffusion, SSLJointDiffusionV2
from .representation_transformer import RepresentationTransformer


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
        self.min_confidence = 0.95
        self.raw_imgs = None
        self.strong_augmenter = torch.nn.Sequential(transforms.RandAugment(magnitude=10))
        self.weak_augmenter = torch.nn.Sequential(transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)))
        if kwargs.get("ckpt_path", None) is not None:
            ignore_keys = kwargs.get("ignore_keys", [])
            only_model = kwargs.get("load_only_unet", False)
            self.init_from_ckpt(kwargs["ckpt_path"], ignore_keys=ignore_keys, only_model=only_model)

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

        if not self.training:
            return loss, loss_dict

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        with torch.no_grad():
            weakly_augmented = self.weak_augmenter(self.raw_imgs)
            encoder_posterior_weak = self.encode_first_stage(weakly_augmented)
            z_weak = self.get_first_stage_encoding(encoder_posterior_weak).detach()

            weak_rep = self.model.diffusion_model.just_representations(
                z_weak, torch.ones(z_weak.shape[0], device=self.device)
            )
            weak_rep = [torch.flatten(z_i, start_dim=1) for z_i in weak_rep]
            weak_rep = torch.concat(weak_rep, dim=1)
            weak_preds = nn.functional.softmax(self.classifier(weak_rep), dim=1).detach()
            pseudo_labels = weak_preds.argmax(dim=1)
            above_threshold_idx ,= (weak_preds.max(dim=1).values > self.min_confidence).nonzero(as_tuple=True)
            pseudo_labels = pseudo_labels[above_threshold_idx]

            if len(above_threshold_idx) == 0:
                return loss, loss_dict

            strongly_augmented = self.strong_augmenter((self.raw_imgs[above_threshold_idx] * 255).type(torch.uint8)) / 255
            strongly_augmented = self.cutout(strongly_augmented, level=1)
            encoder_posterior_strong = self.encode_first_stage(strongly_augmented)
            z_strong = self.get_first_stage_encoding(encoder_posterior_strong).detach()

        strong_rep = self.model.diffusion_model.just_representations(
            z_strong, torch.ones(z_strong.shape[0], device=self.device)
        )
        strong_rep = [torch.flatten(z_i, start_dim=1) for z_i in strong_rep]
        strong_rep = torch.concat(strong_rep, dim=1)
        preds = self.classifier(strong_rep)
        ssl_loss = nn.functional.cross_entropy(preds, pseudo_labels)

        prefix = 'train' if self.training else 'val'
        loss += ssl_loss * len(preds) / len(weak_preds)
        loss_dict.update({f'{prefix}/loss_ssl_classification': ssl_loss})
        loss_dict.update({f'{prefix}/loss': loss})
        accuracy = torch.sum(torch.argmax(preds, dim=1) == pseudo_labels) / len(pseudo_labels)
        loss_dict.update({f'{prefix}/ssl_accuracy': accuracy})

        return loss, loss_dict

    def cutout(self, img_batch, level, fill=0.5):

        """
        Apply cutout to torch tensor of shape (batch, height, width, channel) at the specified level.
        """
        size = 1 + int(level * min(img_batch.shape[1:3]) * 0.499)
        batch, img_height, img_width = img_batch.shape[0:3]
        height_loc = torch.randint(low=0, high=img_height, size=[batch])
        width_loc = torch.randint(low=0, high=img_width, size=[batch])
        x_uppers = (height_loc - size // 2)
        x_uppers *= (x_uppers >= 0)
        x_lowers = (height_loc + size // 2)
        x_lowers -= (x_lowers >= img_height) * (x_lowers - img_height - 1)
        y_uppers = (width_loc - size // 2)
        y_uppers *= (y_uppers >= 0)
        y_lowers = (width_loc + size // 2)
        y_lowers -= (y_lowers >= img_width) * (y_lowers - img_width - 1)

        for img, x_upper, x_lower, y_upper, y_lower in zip(img_batch, x_uppers, x_lowers, y_uppers, y_lowers):
            img[x_upper:x_lower, y_upper:y_lower] = fill
        return img_batch

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.SGD(params, lr=lr, nesterov=True, momentum=0.9, weight_decay=0.0005)
        return opt



class DiffMatchV2(SSLJointDiffusionV2):
    def __init__(
            self,
            first_stage_config,
            cond_stage_config,
            classifier_in_features,
            classifier_hidden,
            num_classes,
            min_confidence=0.95,
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
        self.min_confidence = min_confidence
        self.raw_imgs = None
        self.augmentation = K.augmentation.ImageSequential(K.augmentation.RandomAffine(degrees=0, translate=(0.125, 0.125)))
        self.strong_augmentation = K.augmentation.AugmentationSequential(K.augmentation.auto.RandAugment(n=2, m=10))
        if kwargs.get("ckpt_path", None) is not None:
            ignore_keys = kwargs.get("ignore_keys", [])
            only_model = kwargs.get("load_only_unet", False)
            self.init_from_ckpt(kwargs["ckpt_path"], ignore_keys=ignore_keys, only_model=only_model)

    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False, cond_key=None, return_original_cond=False, bs=None):
        if self.training:
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

        if not self.training:
            return loss, loss_dict

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        prefix = 'train' if self.training else 'val'

        with torch.no_grad():
            weakly_augmented = self.augmentation(self.raw_imgs)
            encoder_posterior_weak = self.encode_first_stage(weakly_augmented)
            z_weak = self.get_first_stage_encoding(encoder_posterior_weak).detach()

            weak_rep = self.model.diffusion_model.just_representations(
                z_weak,
                torch.ones(z_weak.shape[0], device=self.device),
                pooled=False
            )
            weak_rep = self.transform_representations(weak_rep)
            weak_preds = nn.functional.softmax(self.classifier(weak_rep), dim=1).detach()
            pseudo_labels = weak_preds.argmax(dim=1)
            above_threshold_idx ,= (weak_preds.max(dim=1).values > self.min_confidence).nonzero(as_tuple=True)
            pseudo_labels = pseudo_labels[above_threshold_idx]

            loss_dict.update({f'{prefix}/ssl_above_threshold': len(above_threshold_idx) / len(weak_preds)})
            loss_dict.update({f'{prefix}/ssl_max_confidence': weak_preds.max()})
            if len(above_threshold_idx) == 0:
                return loss, loss_dict

            strongly_augmented = self.strong_augmentation((self.raw_imgs[above_threshold_idx]))
            strongly_augmented = self.cutout(strongly_augmented, level=1)
            encoder_posterior_strong = self.encode_first_stage(strongly_augmented)
            z_strong = self.get_first_stage_encoding(encoder_posterior_strong).detach()

        strong_rep = self.model.diffusion_model.just_representations(
            z_strong,
            torch.ones(z_strong.shape[0], device=self.device),
            pooled=False
        )
        strong_rep = self.transform_representations(strong_rep)
        preds = self.classifier(strong_rep)
        ssl_loss = nn.functional.cross_entropy(preds, pseudo_labels)

        loss += ssl_loss * len(preds) / len(weak_preds)
        loss_dict.update({f'{prefix}/loss_ssl_classification': ssl_loss})
        loss_dict.update({f'{prefix}/loss': loss})
        accuracy = torch.sum(torch.argmax(preds, dim=1) == pseudo_labels) / len(pseudo_labels)
        loss_dict.update({f'{prefix}/ssl_accuracy': accuracy})

        return loss, loss_dict

    def cutout(self, img_batch, level, fill=0.5):

        """
        Apply cutout to torch tensor of shape (batch, height, width, channel) at the specified level.
        """
        size = 1 + int(level * min(img_batch.shape[1:3]) * 0.499)
        batch, img_height, img_width = img_batch.shape[0:3]
        height_loc = torch.randint(low=0, high=img_height, size=[batch])
        width_loc = torch.randint(low=0, high=img_width, size=[batch])
        x_uppers = (height_loc - size // 2)
        x_uppers *= (x_uppers >= 0)
        x_lowers = (height_loc + size // 2)
        x_lowers -= (x_lowers >= img_height) * (x_lowers - img_height - 1)
        y_uppers = (width_loc - size // 2)
        y_uppers *= (y_uppers >= 0)
        y_lowers = (width_loc + size // 2)
        y_lowers -= (y_lowers >= img_width) * (y_lowers - img_width - 1)

        for img, x_upper, x_lower, y_upper, y_lower in zip(img_batch, x_uppers, x_lowers, y_uppers, y_lowers):
            img[x_upper:x_lower, y_upper:y_lower] = fill
        return img_batch


class DiffMatchV3(DiffMatchV2):
    def __init__(
            self,
            first_stage_config,
            cond_stage_config,
            attention_config,
            min_confidence=0.95,
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
            min_confidence,
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
