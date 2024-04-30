from typing import Optional
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from einops import repeat
import kornia as K
import kornia.augmentation as aug
from ldm.models.diffusion.ddpm import DDPM
from ldm.modules.diffusionmodules.util import extract_into_tensor, noise_like
from ldm.util import default
from ldm.modules.ema import LitEma
from contextlib import contextmanager
from ..representation_transformer import RepresentationTransformer
from ..adjusted_unet import AdjustedUNet


class JointDiffusionNoisyClassifier(DDPM):
    def __init__(self,
                 unet_config,
                 classifier_in_features,
                 classifier_hidden,
                 num_classes,
                 dropout=0,
                 classification_loss_scale=1,
                 classification_start=0,
                 sample_grad_scale=60,
                 classification_key=1,
                 first_stage_key="image",
                 conditioning_key=None,
                 sampling_method="conditional_to_x",
                 *args,
                 **kwargs):
        super().__init__(
            unet_config,
            first_stage_key=first_stage_key,
            conditioning_key=conditioning_key,
            *args,
            **kwargs
        )
        self.num_classes = num_classes
        self.classification_key = classification_key
        self.classification_loss_scale = classification_loss_scale
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_features, classifier_hidden),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            # nn.ReLU(),
            nn.Linear(classifier_hidden, self.num_classes)
        )
        self.sampling_method = sampling_method
        self.sample_grad_scale = sample_grad_scale
        self.classification_start = classification_start
        self.batch_classes = None
        self.batch_class_predictions = None
        self.sample_classes = None
        # Attributes that will store img labels and labels predictions
        # This is really ugly but since we are unable to change the parent classes and we don't want to copy-paste
        # code (especially that we'd have to copy a lot), this solution seems to be marginally better.
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if kwargs.get("ckpt_path", None) is not None:
            ignore_keys = kwargs.get("ignore_keys", [])
            only_model = kwargs.get("load_only_unet", False)
            self.init_from_ckpt(kwargs["ckpt_path"], ignore_keys=ignore_keys, only_model=only_model)
        self.val_labels = torch.tensor([])
        self.val_preds = torch.tensor([])

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def get_input(self, batch, k):
        self.batch_classes = batch[self.classification_key]
        return super().get_input(batch, k)

    def apply_model(self, x_noisy, t, return_ids=False):
        if hasattr(self, "split_input_params"):
            raise NotImplementedError(
                "This feature is not available for this model")

        x_recon, representations = self.model.diffusion_model(x_noisy, t, pooled=False)
        if isinstance(representations, list): # TODO refactor this shit
            representations = self.transform_representations(representations)
            self.batch_class_predictions = self.classifier(representations)
        else:
            self.batch_class_predictions = representations

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        t = t.cpu()
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        if (self.batch_classes is not None) and (self.classification_start <= 0):
            prefix = 'train' if self.training else 'val'
            self.batch_classes = self.batch_classes if len(self.batch_classes.shape) == 1 else nn.functional.softmax(self.batch_classes, dim=-1)
            loss_classification = nn.functional.cross_entropy(
                self.batch_class_predictions, self.batch_classes)
            loss += loss_classification * self.classification_loss_scale
            loss_dict.update(
                {f'{prefix}/loss_classification': loss_classification})
            loss_dict.update({f'{prefix}/loss': loss})
            self.batch_classes = self.batch_classes if len(self.batch_classes.shape) == 1 else self.batch_classes.argmax(dim=-1)
            accuracy = torch.sum(torch.argmax(
                self.batch_class_predictions, dim=1) == self.batch_classes) / len(self.batch_classes)
            loss_dict.update({f'{prefix}/accuracy': accuracy})
            if prefix == "val":
                self.val_labels = torch.concat([self.val_labels, self.batch_classes.detach().cpu()])
                self.val_preds = torch.concat([self.val_preds, self.batch_class_predictions.argmax(dim=1).detach().cpu()])

        if self.classification_start > 0:
            self.classification_start -= 1
        return loss, loss_dict

    def on_validation_epoch_end(self) -> None:
        wandb.log({"val/conf_mat": wandb.plot.confusion_matrix(
            probs=None,
            y_true=self.val_labels.numpy(),
            preds=self.val_preds.numpy(),
        )})
        self.val_labels = torch.tensor([])
        self.val_preds = torch.tensor([])

    def log_images(self,
                   batch,
                   N=8,
                   n_row=4,
                   sample=True,
                   return_keys=None,
                   sample_classes=None,
                   **kwargs):
        if self.sampling_method == "conditional_to_x":
            self.sample_classes = torch.tensor(sample_classes).to(self.device)
        return super().log_images(
            batch,
            N,
            n_row,
            sample,
            return_keys,
            **kwargs
        )

    def p_mean_variance(self, x, t, clip_denoised: bool):
        if (self.sampling_method == "unconditional") or (self.sample_grad_scale == 0):
            return self.unconditional_p_mean_variance(x, t, clip_denoised)
        elif self.sampling_method == "conditional_to_x":
            return self.grad_guided_p_mean_variance(x, t, clip_denoised)
        elif self.sampling_method == "conditional_to_repr":
            return self.grad_guided_p_mean_variance(x, t, clip_denoised)
        else:
            raise NotImplementedError("Sampling method not implemented")

    def unconditional_p_mean_variance(self, x, t, clip_denoised: bool):
        unet: AdjustedUNet = self.model.diffusion_model
        model_out = unet.just_reconstruction(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def grad_guided_p_mean_variance(self, x, t, clip_denoised: bool):
        if self.sampling_method == "conditional_to_x":
            model_out = self.guided_apply_model(x, t)
        elif self.sampling_method == "conditional_to_repr":
            model_out = self.guided_repr_apply_model(x, t)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def guided_apply_model(self, x, t):
        if not hasattr(self.model.diffusion_model, 'forward_input_blocks'):
            return self.apply_model(x, t)
        t_in = t

        emb = self.model.diffusion_model.get_timestep_embedding(x, t_in, None)

        with torch.enable_grad():
            representations = self.model.diffusion_model.forward_input_blocks(x, None, emb)
            for h in representations:
                h.retain_grad()
            pooled_representations = self.transform_representations(representations)
            class_predictions = self.classifier(pooled_representations)
            # loss = -torch.log(torch.gather(class_predictions, 1, self.sample_classes.unsqueeze(dim=1))).sum()
            loss = nn.functional.cross_entropy(class_predictions, self.sample_classes, reduction="sum")
            loss.backward()
            representations = [(h + self.sample_grad_scale * h.grad).detach() for h in representations]

        model_out = self.model.diffusion_model.forward_output_blocks(x, None, emb, representations)
        return model_out

    def transform_representations(self, representations):
        representations = self.model.diffusion_model.pool_representations(representations)
        representations = [torch.flatten(z_i, start_dim=1)
                           for z_i in representations]
        representations = torch.concat(representations, dim=1)
        return representations


class JointDiffusion(JointDiffusionNoisyClassifier):
    def __init__(self, *args, sampling_recurence_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_recurence_steps = sampling_recurence_steps
        self.x_start = None

    def get_input(self, batch, k):
        out = super().get_input(batch, k)
        self.x_start = out
        return out

    def apply_model(self, x_noisy, t, return_ids=False):
        if hasattr(self, "split_input_params"):
            raise NotImplementedError("This feature is not available for this model")

        x_recon = self.model.diffusion_model.just_reconstruction(x_noisy, t)
        if self.x_start is not None:
            representations = self.model.diffusion_model.just_representations(
                self.x_start,
                torch.zeros(self.x_start.shape[0], device=self.device),
                pooled=False
            )
            if isinstance(representations, list):  # TODO refactor this shit
                representations = self.transform_representations(representations)
                self.batch_class_predictions = self.classifier(representations)
            else:
                self.batch_class_predictions = representations

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    @torch.no_grad()
    def guided_apply_model(self, x: torch.Tensor, t):
        unet: AdjustedUNet = self.model.diffusion_model
        if not hasattr(self.model.diffusion_model, 'forward_input_blocks'):
            return unet.just_reconstruction(x, t)

        with torch.enable_grad():
            x = x.requires_grad_(True)
            pred_noise = unet.just_reconstruction(
                x, t, context=None)
            pred_x_start = (
                (x - extract_into_tensor(
                    self.sqrt_one_minus_alphas_cumprod, t, x.shape
                ) * pred_noise) /
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape)
            )
            representations = unet.just_representations(
                pred_x_start, t, context=None, pooled=False)
            pooled_representations = self.transform_representations(
                representations)
            pred = self.classifier(pooled_representations)

            loss = nn.functional.cross_entropy(
                pred, self.sample_classes, reduction="sum")
            grad = torch.autograd.grad(loss, x)[0]
            s_t = self.sample_grad_scale * extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, (1,))[0]
            model_out = (pred_noise + s_t * grad).detach()

        return model_out

    @torch.no_grad()
    def guided_repr_apply_model(self, x, t):
        if not hasattr(self.model.diffusion_model, 'forward_input_blocks'):
            return self.apply_model(x, t)
        t_in = t

        emb = self.model.diffusion_model.get_timestep_embedding(x, t_in, None)
        unet: AdjustedUNet = self.model.diffusion_model

        with torch.enable_grad():
            representations = unet.forward_input_blocks(x, None, emb)

            pred_noise = unet.forward_output_blocks(x, None, emb, representations)
            pred_x_start = (
                (x - extract_into_tensor(
                    self.sqrt_one_minus_alphas_cumprod, t, x.shape
                ) * pred_noise) /
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape)
            )
            repr_x0 = unet.just_representations(
                pred_x_start, t, context=None, pooled=False)
            pooled_representations = self.transform_representations(
                repr_x0)
            class_predictions = self.classifier(pooled_representations)
            # loss = -torch.log(torch.gather(class_predictions, 1, self.sample_classes.unsqueeze(dim=1))).sum()
            loss = -nn.functional.cross_entropy(class_predictions, self.sample_classes, reduction="sum")
            s_t = self.sample_grad_scale * extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, (1,))[0]
            grads = torch.autograd.grad(loss, representations)
            representations = [(h + self.sample_grad_scale * grad * s_t).detach() for h, grad in zip(representations, grads)]

        model_out = unet.forward_output_blocks(x, None, emb, representations)
        return model_out

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, sample_classes=None, use_ema=True, grad_scales=None, **kwargs):
        if self.sampling_method != "unconditional":
            self.sample_classes = torch.tensor(sample_classes).to(self.device)
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            if grad_scales is None:
                if use_ema is True:
                    with self.ema_scope("Plotting"):
                        samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)
                else:
                    samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

                log["samples"] = samples
                # log["denoise_row"] = self._get_rows_from_list(denoise_row)
            else:
                for grad_scale in grad_scales:
                    self.sample_grad_scale = grad_scale
                    if use_ema is True:
                        with self.ema_scope("Plotting"):
                            samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)
                    else:
                        samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

                    log[f"samples_grad_scale={grad_scale}"] = samples
                    # log[f"denoise_row_grad_scale={grad_scale}"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    @torch.no_grad()
    def p_sample_loop(
        self, shape, return_intermediates=False, x_start=None, t_start=None
    ):
        device = self.betas.device
        b = shape[0]
        img = (
            torch.randn(shape, device=device)
            if x_start is None
            else x_start.clone().to(device)
        )
        num_timesteps = self.num_timesteps if t_start is None else t_start
        intermediates = [img]
        for i in tqdm(
            reversed(range(0, num_timesteps)), desc="Sampling t", total=num_timesteps
        ):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            for _ in range(self.sampling_recurence_steps - 1):
                z_t_minus_1 = self.p_sample(
                    img,
                    t,
                    clip_denoised=self.clip_denoised,
                )
                eps = noise_like(img.shape, img.device)
                alph_t_minus_one = (
                    extract_into_tensor(self.alphas_cumprod, t - 1, (1,))[0]
                    if i != 0
                    else torch.tensor(1, device=img.device, dtype=float)
                )
                alpha_coef = extract_into_tensor(self.alphas_cumprod, t, (1,))[0] / alph_t_minus_one
                img = torch.sqrt(alpha_coef) * z_t_minus_1 + torch.sqrt(1 - alpha_coef) * eps
            img = self.p_sample(
                img,
                t,
                clip_denoised=self.clip_denoised,
            )
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(
        self, batch_size=16, return_intermediates=False, x_start=None, t_start=None
    ):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(
            (batch_size, channels, image_size, image_size),
            return_intermediates=return_intermediates,
            x_start=x_start,
            t_start=t_start,
        )


class JointDiffusionAugmentations(JointDiffusion):
    def __init__(self, *args, augmentations=True, **kwargs):
        super().__init__(*args, **kwargs)
        img_size = self.image_size
        self.augmentation = None
        if augmentations is True:
            self.augmentation = K.augmentation.ImageSequential(
                aug.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.25),
                aug.RandomResizedCrop((img_size, img_size), scale=(0.5, 1), p=0.25),
                aug.RandomRotation((-30, 30), p=0.25),
                aug.RandomHorizontalFlip(0.5),
                aug.RandomContrast((0.6, 1.8), p=0.25),
                aug.RandomSharpness((0.4, 2), p=0.25),
                aug.RandomBrightness((0.6, 1.8), p=0.25),
                aug.RandomMixUpV2(p=0.5),
                # random_apply=(1, 6),
                aug.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            )
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if kwargs.get("ckpt_path", None) is not None:
            ignore_keys = kwargs.get("ignore_keys", [])
            only_model = kwargs.get("load_only_unet", False)
            self.init_from_ckpt(kwargs["ckpt_path"], ignore_keys=ignore_keys, only_model=only_model)

    def get_input(self, batch, k):
        if self.augmentation is not None:
            out = super().get_input(batch, k)
            if self.training:
                self.x_start = self.augmentation(out)
            else:
                self.x_start = out
            return out
        else:
            if self.training:
                self.batch_classes = batch[self.classification_key]
                x = batch[k]
                self.x_start = x[1]
                return x[0]
            else:
                x = batch[k]
                if type(batch[k]) in (list, tuple):
                    x = x[0]
                self.batch_classes = batch[self.classification_key]
                self.x_start = x
                return x


class JointDiffusionAttention(JointDiffusionAugmentations):
    def __init__(self, attention_config, *args, **kwargs):
        super().__init__(
            classifier_in_features=0,
            classifier_hidden=0,
            num_classes=0,
            *args,
            **kwargs
        )
        self.classifier = RepresentationTransformer(**attention_config)

        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if kwargs.get("ckpt_path", None) is not None:
            ignore_keys = kwargs.get("ignore_keys", [])
            only_model = kwargs.get("load_only_unet", False)
            self.init_from_ckpt(kwargs["ckpt_path"], ignore_keys=ignore_keys, only_model=only_model)

    def transform_representations(self, representations):
        return representations


class JointDiffusionAttentionDoubleOptims(JointDiffusionAttention):
    def __init__(self, attention_config, *args, classifier_lr=0.01, **kwargs):
        super().__init__(attention_config, *args, **kwargs)
        self.automatic_optimization = False
        self.learning_rate_classifier = classifier_lr

    def configure_optimizers(self):
        lr_diffusion = self.learning_rate
        params_diffusion = list(self.model.parameters())
        opt_diffusion = torch.optim.AdamW(params_diffusion, lr=lr_diffusion)

        lr_classifier = self.learning_rate_classifier
        params_classifier = list(self.classifier.parameters())
        opt_classifier = torch.optim.AdamW(params=params_classifier, lr=lr_classifier)
        return opt_diffusion, opt_classifier

    def training_step(self, batch, batch_idx):
        opt_diffusion, opt_classifier = self.optimizers()
        opt_diffusion.zero_grad()
        opt_classifier.zero_grad()
        loss = super().training_step(batch, batch_idx)
        self.manual_backward(loss)
        opt_diffusion.step()
        opt_classifier.step()


# class JointDiffusion(JointDiffusionAugmentations):
#     def __init__(self, old_model, new_model, old_classes, new_classes, *args, kd_loss_weight=1.0, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.old_model = old_model
#         self.old_classes = torch.tensor(old_classes, device=self.device)
#         self.new_model = new_model
#         self.new_classes = torch.tensor(new_classes, device=self.device)
#         self.kd_loss_weight = kd_loss_weight
#         self.x_recon = None

#     def apply_model(self, x_noisy, t, return_ids=False):
#         if hasattr(self, "split_input_params"):
#             raise NotImplementedError("This feature is not available for this model")

#         x_recon = self.model.diffusion_model.just_reconstruction(x_noisy, t)
#         if self.x_start is not None:
#             representations = self.model.diffusion_model.just_representations(
#                 self.x_start,
#                 torch.zeros(self.x_start.shape[0], device=self.device),
#                 pooled=False
#             )
#             if isinstance(representations, list):  # TODO refactor this shit
#                 representations = self.transform_representations(representations)
#                 self.batch_class_predictions = self.classifier(representations)
#             else:
#                 self.batch_class_predictions = representations

#         if isinstance(x_recon, tuple) and not return_ids:
#             self.x_recon = x_recon[0]
#             return x_recon[0]
#         else:
#             self.x_recon = x_recon
#             return x_recon

#     def p_losses(self, x_start, t, noise=None):
#         loss, loss_dict = super().p_losses(x_start, t, noise)
#         old_classes_mask = torch.any(self.batch_classes.unsqueeze(-1) == self.old_classes, dim=-1)
#         new_classes_mask = torch.any(self.batch_classes.unsqueeze(-1) == self.new_classes, dim=-1)

#         # diffusion knowledge distillation

#         # classifier knowledge distillation
