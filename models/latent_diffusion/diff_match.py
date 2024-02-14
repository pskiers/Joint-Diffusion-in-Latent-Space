import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange
import kornia as K
from .ssl_joint_diffusion import SSLJointLatentDiffusion, SSLJointLatentDiffusionV2
from .joint_latent_diffusion import JointLatentDiffusion, JointLatentDiffusionAttention
from ..representation_transformer import RepresentationTransformer
from ..adjusted_unet import AdjustedUNet
from ..ddim import DDIMSamplerGradGuided
from ..utils import FixMatchEma, interleave, de_interleave


class LatentDiffMatch(SSLJointLatentDiffusion):
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


class LatentDiffMatchV2(SSLJointLatentDiffusionV2):
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


class LatentDiffMatchV3(LatentDiffMatchV2):
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


class LatentDiffMatchWithSampling(LatentDiffMatchV3):
    def __init__(
            self,
            first_stage_config,
            cond_stage_config,
            attention_config,
            noisy_attention_config,
            generation_start,
            generation_batch_size,
            num_classes,
            step_per_generation=5,
            ddim_steps=20,
            min_confidence=0.95,
            classification_loss_scale=1,
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
            attention_config,
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
        self.num_classes = num_classes
        self.ddim_steps = ddim_steps
        self.noisy_classifier = RepresentationTransformer(**noisy_attention_config)
        self.gradient_guided_sampling = True
        self.noisy_class_predictions = None
        self.generation_start = generation_start
        self.generation_batch = generation_batch_size
        self.step_per_generation = step_per_generation
        self.current_generation_step = step_per_generation
        if kwargs.get("ckpt_path", None) is not None:
            ignore_keys = kwargs.get("ignore_keys", [])
            only_model = kwargs.get("load_only_unet", False)
            self.init_from_ckpt(kwargs["ckpt_path"], ignore_keys=ignore_keys, only_model=only_model)

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if hasattr(self, "split_input_params"):
            raise NotImplementedError("This feature is not available for this model")

        x_recon, noisy_rep = self.model.diffusion_model(x_noisy, t, pooled=False)
        noisy_rep = self.transform_representations(noisy_rep)
        self.noisy_class_predictions = self.classifier(noisy_rep)
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

    def p_losses(self, x_start, cond, t, noise=None):
        loss, loss_dict = super().p_losses(x_start, cond, t, noise)

        prefix = 'train' if self.training else 'val'
        if self.batch_classes is not None:

            noisy_loss_classification = nn.functional.cross_entropy(
                self.noisy_class_predictions, self.batch_classes)
            loss += noisy_loss_classification
            loss_dict.update(
                {f'{prefix}/loss_noisy_classification': noisy_loss_classification})
            loss_dict.update({f'{prefix}/loss': loss})
            accuracy = torch.sum(torch.argmax(
                self.noisy_class_predictions, dim=1) == self.batch_classes) / len(self.batch_classes)
            loss_dict.update({f'{prefix}/noisy_accuracy': accuracy})

        if self.generation_start <= 0 and self.training:
            if self.current_generation_step >= self.step_per_generation:
                self.current_generation_step = 0
                prev_classes = self.sample_classes
                self.sample_classes = torch.randint(
                    0,
                    self.num_classes,
                    size=[self.generation_batch * self.step_per_generation],
                    device=self.device
                )
                self.generation_classes = self.sample_classes
                # generated = self.sample(cond=None, batch_size=self.generation_batch)

                ddim_sampler = DDIMSamplerGradGuided(self)
                shape = (self.channels, self.image_size, self.image_size)
                self.generated, _ = ddim_sampler.sample(
                    self.ddim_steps, self.generation_batch * self.step_per_generation, shape, cond=None, verbose=False)
                self.sample_classes = prev_classes

            start = self.generation_batch * self.current_generation_step
            end = self.generation_batch * (self.current_generation_step + 1)
            generated = self.generated[start:end]
            labels = self.generation_classes[start:end]
            self.current_generation_step += 1
            generated_rep = self.model.diffusion_model.just_representations(
                generated,
                torch.ones(generated.shape[0], device=self.device),
                pooled=False
            )
            generated_rep = self.transform_representations(generated_rep)
            preds = self.classifier(generated_rep)
            gen_loss = nn.functional.cross_entropy(preds, labels)

            loss += gen_loss
            loss_dict.update(
                {f'{prefix}/loss_generation_classification': gen_loss})
            loss_dict.update({f'{prefix}/loss': loss})
            gen_accuracy = torch.sum(torch.argmax(
                preds, dim=1) == labels) / len(labels)
            loss_dict.update({f'{prefix}/generation_accuracy': gen_accuracy})

        else:
            if self.training:
                self.generation_start -= 1
        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        self.noisy_classifier, self.classifier = self.classifier, self.noisy_classifier
        if self.gradient_guided_sampling is True:
            out = self.grad_guided_p_mean_variance(x, c, t, clip_denoised, return_codebook_ids, quantize_denoised,
                                                    return_x0, score_corrector, corrector_kwargs)
        else:
            out = super().p_mean_variance(x, c, t, clip_denoised, return_codebook_ids, quantize_denoised,
                                           return_x0, score_corrector, corrector_kwargs)
        self.noisy_classifier, self.classifier = self.classifier, self.noisy_classifier
        return out


class LatentDiffMatchPooling(JointLatentDiffusion):
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 classifier_in_features,
                 classifier_hidden,
                 num_classes,
                 min_confidence=0.95,
                 mu=7,
                 batch_size=64,
                 classification_start=0,
                 dropout=0,
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
            first_stage_config,
            cond_stage_config,
            classifier_in_features,
            classifier_hidden,
            num_classes,
            dropout,
            classification_loss_weight,
            classification_key,
            False,
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
        self.mu = mu
        self.batch_size = batch_size
        self.classification_start = classification_start

    def get_input(self,
                  batch,
                  k,
                  return_first_stage_outputs=False,
                  force_c_encode=False,
                  cond_key=None,
                  return_original_cond=False,
                  bs=None):
        if self.training is True:
            batch = batch[1][0]
        batch[k] = rearrange(batch[k], 'b c h w -> b h w c')
        return super().get_input(
            batch,
            k,
            return_first_stage_outputs,
            force_c_encode,
            cond_key,
            return_original_cond,
            bs
        )

    def get_train_classification_input(self, batch, k):
        x = batch[0][k]
        x = self.to_latent(x, arrange=False)

        y = batch[0][self.classification_key]

        _, weak_img, strong_img = batch[1][0]
        weak_img = self.to_latent(weak_img, arrange=False)
        strong_img = self.to_latent(strong_img, arrange=False)

        return x, y, weak_img, strong_img

    def get_valid_classification_input(self, batch, k):
        x = batch[k]
        x = self.to_latent(x, arrange=False)
        y = batch[self.classification_key]
        return x, y

    def train_classification_step(self, batch, loss):
        if self.classification_start > 0:
            self.classification_start -= 1
            return loss

        loss_dict = {}
        x, y, weak_img, strong_img = self.get_train_classification_input(
            batch, self.first_stage_key)
        t = torch.zeros((x.shape[0]*(2*self.mu+1),), device=self.device).long()

        inputs = interleave(
            torch.cat((x, weak_img, strong_img)), 2*self.mu+1)

        unet: AdjustedUNet = self.model.diffusion_model
        representations = unet.just_representations(inputs, t)
        representations = self.transform_representations(representations)
        logits = self.classifier(representations)

        logits = de_interleave(logits, 2*self.mu+1)
        preds_x = logits[:self.batch_size]
        preds_weak, preds_strong = logits[self.batch_size:].chunk(2)
        del logits

        loss_classification = nn.functional.cross_entropy(
            preds_x, y, reduction="mean")
        loss += loss_classification * self.classification_loss_weight
        accuracy = torch.sum(torch.argmax(preds_x, dim=1) == y) / len(y)
        loss_dict.update(
            {'train/loss_classification': loss_classification})
        loss_dict.update({'train/loss': loss})
        loss_dict.update({'train/accuracy': accuracy})

        pseudo_label = torch.softmax(preds_weak.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.min_confidence).float()
        ssl_loss = (nn.functional.cross_entropy(
            preds_strong, targets_u, reduction='none') * mask).mean()
        loss += ssl_loss * self.classification_loss_weight
        accuracy = torch.sum(
            (torch.argmax(preds_strong, dim=1) == targets_u) * mask
        ) / mask.sum() if mask.sum() > 0 else 0
        loss_dict.update(
            {'train/ssl_above_threshold': mask.mean().item()})
        loss_dict.update({'train/ssl_max_confidence': mask.max().item()})
        loss_dict.update({'train/loss_ssl_classification': ssl_loss})
        loss_dict.update({'train/loss': loss})
        loss_dict.update({'train/ssl_accuracy': accuracy})

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        return loss


class LatentDiffMatchAttention(LatentDiffMatchPooling):
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 attention_config,
                 min_confidence=0.95,
                 mu=7,
                 batch_size=64,
                 classification_start=0,
                 classification_loss_weight=1,
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
        self._init(
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
        self.classification_loss_weight = classification_loss_weight
        self.classification_key = classification_key
        self.gradient_guided_sampling = False
        self.augmentations = None
        self.min_confidence = min_confidence
        self.mu = mu
        self.batch_size = batch_size
        self.classification_start = classification_start
        self.classifier = RepresentationTransformer(**attention_config)

        self.init_ema()
        self.init_ckpt(**kwargs)

    def transform_representations(self, representations):
        return representations
