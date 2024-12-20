from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from ..stylegan_classifier import StyleGANDiscriminator


class JointDiffusionNoisyClassifier(DDPM):
    def __init__(
        self,
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
        **kwargs,
    ):
        super().__init__(
            unet_config, first_stage_key=first_stage_key, conditioning_key=conditioning_key, *args, **kwargs
        )
        self.num_classes = num_classes
        self.classification_key = classification_key
        self.classification_loss_scale = classification_loss_scale
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_features, classifier_hidden),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            # nn.ReLU(),
            nn.Linear(classifier_hidden, self.num_classes),
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
            raise NotImplementedError("This feature is not available for this model")

        x_recon, representations = self.model.diffusion_model(x_noisy, t, pooled=False)
        if isinstance(representations, list):  # TODO refactor this shit
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
        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        t = t.cpu()
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        if (self.batch_classes is not None) and ((self.classification_start <= 0) or not self.training):
            prefix = "train" if self.training else "val"
            self.batch_classes = (
                self.batch_classes
                if len(self.batch_classes.shape) == 1
                else nn.functional.softmax(self.batch_classes, dim=-1)
            )
            loss_classification = nn.functional.cross_entropy(self.batch_class_predictions, self.batch_classes)
            loss += loss_classification * self.classification_loss_scale
            loss_dict.update({f"{prefix}/loss_classification": loss_classification})
            loss_dict.update({f"{prefix}/loss": loss})
            self.batch_classes = (
                self.batch_classes if len(self.batch_classes.shape) == 1 else self.batch_classes.argmax(dim=-1)
            )
            accuracy = torch.sum(torch.argmax(self.batch_class_predictions, dim=1) == self.batch_classes) / len(
                self.batch_classes
            )
            loss_dict.update({f"{prefix}/accuracy": accuracy})
            if prefix == "val":
                self.val_labels = torch.concat([self.val_labels, self.batch_classes.detach().cpu()])
                self.val_preds = torch.concat(
                    [self.val_preds, self.batch_class_predictions.argmax(dim=1).detach().cpu()]
                )

        if self.classification_start > 0:
            self.classification_start -= 1
        return loss, loss_dict

    def on_validation_epoch_end(self) -> None:
        wandb.log(
            {
                "val/conf_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=self.val_labels.numpy(),
                    preds=self.val_preds.numpy(),
                )
            }
        )
        self.val_labels = torch.tensor([])
        self.val_preds = torch.tensor([])

    def log_images(self, batch, N=8, n_row=4, sample=True, return_keys=None, sample_classes=None, **kwargs):
        if self.sampling_method == "conditional_to_x":
            self.sample_classes = torch.tensor(sample_classes).to(self.device)
        return super().log_images(batch, N, n_row, sample, return_keys, **kwargs)

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
            x_recon.clamp_(-1.0, 1.0)

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
            x_recon.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def guided_apply_model(self, x, t):
        if not hasattr(self.model.diffusion_model, "forward_input_blocks"):
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
        representations = [torch.flatten(z_i, start_dim=1) for z_i in representations]
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
                self.x_start, torch.zeros(self.x_start.shape[0], device=self.device), pooled=False
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
        if not hasattr(self.model.diffusion_model, "forward_input_blocks"):
            return unet.just_reconstruction(x, t)

        with torch.enable_grad():
            x = x.requires_grad_(True)
            pred_noise = unet.just_reconstruction(x, t, context=None)
            pred_x_start = (
                x - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * pred_noise
            ) / extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape)
            representations = unet.just_representations(pred_x_start, t, context=None, pooled=False)
            pooled_representations = self.transform_representations(representations)
            pred = self.classifier(pooled_representations)

            loss = nn.functional.cross_entropy(pred, self.sample_classes, reduction="sum")
            grad = torch.autograd.grad(loss, x)[0]
            s_t = self.sample_grad_scale * extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, (1,))[0]
            model_out = (pred_noise + s_t * grad).detach()

        return model_out

    @torch.no_grad()
    def guided_repr_apply_model(self, x, t):
        if not hasattr(self.model.diffusion_model, "forward_input_blocks"):
            return self.apply_model(x, t)
        t_in = t

        emb = self.model.diffusion_model.get_timestep_embedding(x, t_in, None)
        unet: AdjustedUNet = self.model.diffusion_model

        with torch.enable_grad():
            representations = unet.forward_input_blocks(x, None, emb)

            pred_noise = unet.forward_output_blocks(x, None, emb, representations)
            pred_x_start = (
                x - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * pred_noise
            ) / extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape)
            repr_x0 = unet.just_representations(pred_x_start, t, context=None, pooled=False)
            pooled_representations = self.transform_representations(repr_x0)
            class_predictions = self.classifier(pooled_representations)
            # loss = -torch.log(torch.gather(class_predictions, 1, self.sample_classes.unsqueeze(dim=1))).sum()
            loss = -nn.functional.cross_entropy(class_predictions, self.sample_classes, reduction="sum")
            s_t = self.sample_grad_scale * extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, (1,))[0]
            grads = torch.autograd.grad(loss, representations)
            representations = [
                (h + self.sample_grad_scale * grad * s_t).detach() for h, grad in zip(representations, grads)
            ]

        model_out = unet.forward_output_blocks(x, None, emb, representations)
        return model_out

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=8,
        n_row=2,
        sample=True,
        return_keys=None,
        sample_classes=None,
        use_ema=True,
        grad_scales=None,
        **kwargs,
    ):
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
                t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
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

    # @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False, x_start=None, t_start=None):
        device = self.betas.device
        b = shape[0]
        self.lyapunov_stuff = []
        img = torch.randn(shape, device=device, requires_grad=True) if x_start is None else x_start.clone().to(device)
        # img = torch.randn(shape, device=device) if x_start is None else x_start.clone().to(device)
        num_timesteps = self.num_timesteps if t_start is None else t_start
        intermediates = [img.detach()]
        for i in tqdm(reversed(range(0, num_timesteps)), desc="Sampling t", total=num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            if i % 1 == 0:
                with torch.no_grad():
                    def single_forward(x_):
                        return self.p_sample(x_, t, clip_denoised=self.clip_denoised)

                    jacobian = torch.autograd.functional.jacobian(single_forward, img)
                    # jacobian = torch.func.jacrev(single_forward)(img)  # Jacobian as a matrix
                    det = torch.det(jacobian.reshape(2, 2)) 
                    # det = torch.log(torch.linalg.matrix_norm(jacobian.reshape(3*32*32, 3*32*32), ord=2))
                self.lyapunov_stuff.append(det.detach())
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
            # lyap_img = img * 128
            # lyap_img.retain_grad()
            # img_2 = self.p_sample(
            #     lyap_img / 128,
            #     t,
            #     clip_denoised=self.clip_denoised,
            # )
            # torch.sum(img_2 * 128).backward()
            # self.lyapunov_stuff.append(torch.log(torch.sqrt(torch.sum(lyap_img.grad**2, dim=(1, 2, 3)))).detach())

            # v = torch.randn_like(img_2)
            # vjp = torch.autograd.grad(
            #     outputs=img_2,
            #     inputs=img,
            #     grad_outputs=v,
            #     create_graph=True,
            # )[0]
            # self.lyapunov_stuff.append(torch.log(torch.sqrt(torch.sum(vjp**2, dim=(1, 2, 3)))).detach())
            img = img.detach().requires_grad_(True)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    # @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False, x_start=None, t_start=None):
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
        super().__init__(classifier_in_features=0, classifier_hidden=0, num_classes=0, *args, **kwargs)
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


class JointDiffusionKnowledgeDistillation(JointDiffusionAugmentations):
    def __init__(
        self,
        old_model,
        new_model,
        old_classes,
        new_classes,
        *args,
        kd_loss_weight=1.0,
        kl_classification_weight=0.001,
        no_wait_kl_classification=False,
        old_samples_weight=1.0,
        new_samples_weight=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.old_model = old_model
        self.old_classes = torch.tensor(old_classes, device=self.device)
        self.new_model = new_model
        self.new_classes = torch.tensor(new_classes, device=self.device)
        self.kd_loss_weight = kd_loss_weight
        self.kl_classification_weight = kl_classification_weight
        self.model_pred = None
        self.x_noisy = None
        self.old_samples_weight = old_samples_weight
        self.new_samples_weight = new_samples_weight

        self.classes_per_task = len(new_classes)
        self.no_wait_kl_classification = no_wait_kl_classification

    def named_parameters(self, recurse: bool = True):
        named_parameters = super().named_parameters(recurse=recurse)
        named_parameters = [
            (name, param) for name, param in named_parameters if "new_model" not in name and "old_model" not in name
        ]
        return named_parameters

    def parameters(self, recurse: bool = True):
        parameters = super().parameters(recurse=recurse)
        parameters = [param for param in parameters if param is not self.new_model and param is not self.old_model]
        return parameters

    def apply_model(self, x_noisy, t, return_ids=False):
        if hasattr(self, "split_input_params"):
            raise NotImplementedError("This feature is not available for this model")

        self.x_noisy = x_noisy
        model_pred = self.model.diffusion_model.just_reconstruction(x_noisy, t)
        if self.x_start is not None:
            representations = self.model.diffusion_model.just_representations(
                self.x_start, torch.zeros(self.x_start.shape[0], device=self.device), pooled=False
            )
            if isinstance(representations, list):  # TODO refactor this shit
                representations = self.transform_representations(representations)
                self.batch_class_predictions = self.classifier(representations)
            else:
                self.batch_class_predictions = representations

        if isinstance(model_pred, tuple) and not return_ids:
            self.model_pred = model_pred[0]
            return model_pred[0]
        else:
            self.model_pred = model_pred
            return model_pred

    def p_losses(self, x_start, t, noise=None):
        prefix = "train" if self.training else "val"

        # old and new data masks
        old_unet = self.old_model.model.diffusion_model
        new_unet = None if self.new_model is None else self.new_model.model.diffusion_model
        old_classes_mask = torch.any(self.batch_classes.unsqueeze(-1) == self.old_classes.to(self.device), dim=-1)
        new_classes_mask = torch.any(self.batch_classes.unsqueeze(-1) == self.new_classes.to(self.device), dim=-1)
        batch = old_classes_mask.sum() + new_classes_mask.sum()

        #
        if self.new_model is not None:
            loss, loss_dict = super().p_losses(x_start, t, noise)
        else:
            loss, loss_dict = super().p_losses(
                x_start=x_start[new_classes_mask],
                t=t[new_classes_mask],
                noise=None if noise is None else noise[new_classes_mask],
            )
            super().p_losses(x_start, t, noise)
        # diffusion knowledge distillation
        if self.kd_loss_weight > 0:
            loss_old = 0
            if old_classes_mask.sum() != 0:
                with torch.no_grad():
                    old_outputs = old_unet.just_reconstruction(self.x_noisy[old_classes_mask], t[old_classes_mask])
                loss_old = self.get_loss(self.model_pred[old_classes_mask], old_outputs, mean=True)
            loss_new = 0
            if new_classes_mask.sum() != 0 and self.new_model is not None:
                with torch.no_grad():
                    new_outputs = new_unet.just_reconstruction(self.x_noisy[new_classes_mask], t[new_classes_mask])
                loss_new = self.get_loss(self.model_pred[new_classes_mask], new_outputs, mean=True)
            kd_loss = self.kd_loss_weight * (
                loss_new * 2 * (new_classes_mask.sum() / batch) * self.new_samples_weight
                + loss_old * 2 * (old_classes_mask.sum() / batch) * self.old_samples_weight
            )
            loss += kd_loss
            loss_dict.update({f"{prefix}/loss_diffusion_kd": kd_loss})

        if self.classification_start <= 0 or self.no_wait_kl_classification and self.kl_classification_weight > 0:
            # classifier knowledge distillation
            loss_old = 0
            if old_classes_mask.sum() != 0:
                with torch.no_grad():
                    old_repr = old_unet.just_representations(
                        self.x_start[old_classes_mask],
                        torch.zeros(self.x_start[old_classes_mask].shape[0], device=self.device),
                        pooled=False,
                    )
                    if isinstance(old_repr, list):  # TODO refactor this shit
                        old_repr = self.old_model.transform_representations(old_repr)
                        old_preds = self.old_model.classifier(old_repr)
                    else:
                        old_preds = old_repr
                    old_preds = nn.functional.softmax(old_preds, dim=1).detach()
                loss_old = nn.functional.cross_entropy(self.batch_class_predictions[old_classes_mask], old_preds)

            loss_new = 0
            if new_classes_mask.sum() != 0 and self.new_model is not None:
                with torch.no_grad():
                    new_repr = new_unet.just_representations(
                        self.x_start[new_classes_mask],
                        torch.zeros(self.x_start[new_classes_mask].shape[0], device=self.device),
                        pooled=False,
                    )
                    if isinstance(new_repr, list):  # TODO refactor this shit
                        new_repr = self.new_model.transform_representations(new_repr)
                        new_preds = self.new_model.classifier(new_repr)
                    else:
                        new_preds = new_repr
                    new_preds = nn.functional.softmax(new_preds, dim=1).detach()
                loss_new = nn.functional.cross_entropy(self.batch_class_predictions[new_classes_mask], new_preds)

            kd_loss = (
                self.kl_classification_weight
                * self.kd_loss_weight
                * (
                    loss_new * 2 * (new_classes_mask.sum() / batch) * self.new_samples_weight
                    + loss_old * 2 * (old_classes_mask.sum() / batch) * self.old_samples_weight
                )
            )
            loss += kd_loss
            loss_dict.update({f"{prefix}/loss_classifier_kd": kd_loss})
            loss_dict.update({f"{prefix}/loss": loss})

        # per class accuracy logging
        for i in range(len(self.old_classes) // self.classes_per_task):
            task_classes = self.old_classes[i * self.classes_per_task : (i + 1) * self.classes_per_task]
            task_mask = torch.any(self.batch_classes.unsqueeze(-1) == task_classes.to(self.device), dim=-1)
            curr_task_acc = torch.sum(
                torch.argmax(self.batch_class_predictions[task_mask], dim=1) == self.batch_classes[task_mask]
            ) / len(self.batch_classes[task_mask])
            loss_dict.update({f"cl_tasks_{prefix}/task{i}_accuracy": curr_task_acc})
        i += 1
        curr_task_acc = torch.sum(
            torch.argmax(self.batch_class_predictions[new_classes_mask], dim=1) == self.batch_classes[new_classes_mask]
        ) / len(self.batch_classes[new_classes_mask])
        loss_dict.update({f"cl_tasks_{prefix}/task{i}_accuracy": curr_task_acc})
        return loss, loss_dict


class JointDiffusionAdversarialKnowledgeDistillation(JointDiffusionKnowledgeDistillation):
    def __init__(
        self,
        num_classes,
        repr_sizes: List[int],
        disc_lr: float,
        classifier_lr: float,
        disc_channel_div: int = 1,
        adv_loss_scale: float = 1.0,
        renoised_classification_loss_scale: float = 0,
        disc_input_mode: str = "x0_renoised",
        disc_use_soft_labels: bool = False,
        renoised_classification_threshold: float = 1.0,
        renoised_classification_min_t: int = 100,
        renoised_classification_max_t: int = 900,
        emb_proj: Optional[int] = None,
        renoiser_mean: float = 1.0,
        renoiser_std: float = 1.0,
        start_from_mean_weights: bool = True,
        accumulate_grad_batches: int = 1,
        use_old_and_new: bool = True,
        disc_pass_x_noisy: bool = False,
        grad_clip_val: float = 0.5,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(num_classes=num_classes, *args, **kwargs)
        self.emb_proj = emb_proj
        self.disc_pass_x_noisy = disc_pass_x_noisy
        if self.disc_pass_x_noisy:
            repr_sizes = [repr_size * 2 for repr_size in repr_sizes]
        self.new_disc = StyleGANDiscriminator(
            context_dims=repr_sizes, projection_div=disc_channel_div, emb_proj=emb_proj, emb_dim=num_classes
        )
        self.old_disc = StyleGANDiscriminator(
            context_dims=repr_sizes, projection_div=disc_channel_div, emb_proj=emb_proj, emb_dim=num_classes
        )
        self.adv_loss_scale = adv_loss_scale
        self.renoised_classification_loss_scale = renoised_classification_loss_scale
        self.renoised_classification_threshold = renoised_classification_threshold
        self.renoised_classification_min_t = renoised_classification_min_t
        self.renoised_classification_max_t = renoised_classification_max_t
        self.disc_lr = disc_lr
        self.classifier_lr = classifier_lr
        self.automatic_optimization = False
        self.grad_clip_val = grad_clip_val
        self.phase = "student"
        assert disc_input_mode in ["x0", "x_t-1", "x0_renoised", "x0_no_ladd", "x0_renoised_no_ladd"]
        self.disc_input_mode = disc_input_mode
        self.disc_use_soft_labels = disc_use_soft_labels
        self.accumulate_grad_batches = accumulate_grad_batches

        self.use_old_and_new = use_old_and_new
        self.renoiser_mean = renoiser_mean
        self.renoiser_std = renoiser_std
        self.renoiser_distribution = torch.distributions.Normal(renoiser_mean, renoiser_std)

        if start_from_mean_weights and self.old_model is not None and self.new_model is not None:
            for param, param_old, param_new in zip(
                self.parameters(), self.old_model.parameters(), self.new_model.parameters()
            ):
                param.data = (param_old.data + param_new.data) / 2.0
            for param, param_old, param_new in zip(
                self.classifier[-1].parameters(),
                self.old_model.classifier[-1].parameters(),
                self.new_model.classifier[-1].parameters(),
            ):
                param.data[: len(self.old_classes)] = param_old.data[: len(self.old_classes)]
                param.data[len(self.old_classes) : len(self.old_classes) + len(self.new_classes)] = param_new.data[
                    len(self.old_classes) : len(self.old_classes) + len(self.new_classes)
                ]
                param.data[len(self.old_classes) + len(self.new_classes) :] = param_old.data[
                    len(self.old_classes) + len(self.new_classes) :
                ]

        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    def sample_renoise_timestep(self, shape: torch.Size) -> torch.Tensor:
        norm_samples = self.renoiser_distribution.rsample(shape).to(self.device)
        logitnormal = torch.sigmoid(norm_samples)
        return (logitnormal * self.num_timesteps).long()

    def configure_optimizers(self) -> Tuple[torch.optim.AdamW, torch.optim.AdamW, torch.optim.AdamW]:
        params_diffusion = list(self.model.parameters())
        opt_diffusion = torch.optim.AdamW(params_diffusion, lr=self.learning_rate)

        params_classifier = list(self.classifier.parameters())
        opt_classifier = torch.optim.AdamW(params=params_classifier, lr=self.classifier_lr)

        params_disc = list(self.new_disc.parameters()) + list(self.old_disc.parameters())
        opt_disc = torch.optim.AdamW(params=params_disc, lr=self.disc_lr)

        return opt_diffusion, opt_classifier, opt_disc

    def training_step(self, batch, batch_idx) -> None:
        opt_diffusion, opt_classifier, opt_disc = self.optimizers()
        self.phase = "student" if (batch_idx // self.accumulate_grad_batches) % 2 == 0 else "disc"
        loss = super().training_step(batch, batch_idx)
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_val)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            if self.phase == "student":
                opt_disc.zero_grad(set_to_none=True)
                opt_classifier.zero_grad(set_to_none=True)
                opt_diffusion.step()
            elif self.phase == "disc":
                opt_disc.step()
                opt_classifier.step()
                opt_diffusion.zero_grad(set_to_none=True)
            opt_diffusion.zero_grad()
            opt_classifier.zero_grad()
            opt_disc.zero_grad()

    def sample_xt_1(self, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True) -> torch.Tensor:
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.unconditional_p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def predict_x0(
        self, x_noisy: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor, clip_denoised: bool = False
    ) -> torch.Tensor:
        x0_pred = (
            x_noisy
            - extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod,
                t,
                x_noisy.shape,
            )
            * pred_noise
        ) / extract_into_tensor(self.sqrt_alphas_cumprod, t, x_noisy.shape)
        if clip_denoised:
            x0_pred.clamp_(-1.0, 1.0)
        return x0_pred

    def discriminate(
        self,
        disc: nn.Module,
        unet: AdjustedUNet,
        x_B_C_W_H: torch.Tensor,
        t: torch.Tensor,
        emb: Optional[torch.Tensor],
        x_noisy: torch.Tensor,
    ) -> torch.Tensor:
        if self.disc_pass_x_noisy:
            x_in = torch.concat([x_B_C_W_H, x_noisy], dim=0)
            t_in = torch.concat([t, t], dim=0)
            repr_in = unet.just_representations(x_in, timesteps=t_in, pooled=False)
            repr_in = [torch.concat(r.chunk(2), dim=1) for r in repr_in]
            out = disc(repr_in, emb=emb)
        else:
            repr_in = unet.just_representations(x_B_C_W_H, timesteps=t, pooled=False)
            out = disc(repr_in, emb=emb)
        return out

    def get_adversarial_loss(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        pred_noise: torch.Tensor,
        x_start: torch.Tensor,
        classes: torch.Tensor,
        repr_transformer: Callable,
        unet: AdjustedUNet,
        disc: nn.Module,
        classifier: nn.Module,
        loss_dict: dict,
        prefix: str = "train",
        suffix: str = "old",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.emb_proj is None:
            emb = None
        elif self.disc_use_soft_labels:
            with torch.no_grad():
                reprs = unet.just_representations(
                    x_start,
                    torch.zeros(x_start.shape[0], device=self.device),
                    pooled=False,
                )
                if isinstance(reprs, list):  # TODO refactor this shit
                    reprs = repr_transformer(reprs)
                    classes = classifier(reprs)
                else:
                    classes = reprs
                classes = nn.functional.softmax(classes, dim=1).detach()
            emb = classes
        else:
            emb = F.one_hot(classes, self.num_classes).float()
        x_false, t_false, x_true, t_true, x0_pred = self.get_adversial_inputs(x_noisy, t, pred_noise, x_start, unet)
        out_false = self.discriminate(disc, unet, x_false, t_false, emb, x_noisy)

        if self.phase == "student":
            loss = -out_false.sum(dim=1).mean()
            if self.renoised_classification_loss_scale > 0:
                t_mask = (t > self.renoised_classification_min_t) & (t < self.renoised_classification_max_t)
                repr = unet.just_representations(x0_pred, timesteps=torch.zeros_like(t_true), pooled=False)
                repr = self.transform_representations(repr)
                logits = classifier(repr)
                loss_renoised_classification = F.cross_entropy(logits, classes, reduction="none")
                loss_renoised_classification = loss_renoised_classification * t_mask
                if self.renoised_classification_threshold < 1.0:
                    indices = classes if len(classes.shape) == 1 else classes.argmax(dim=1)
                    mask = (
                        torch.gather(F.softmax(logits), dim=1, index=indices.unsqueeze(-1)).squeeze(-1)
                        > self.renoised_classification_threshold
                    )
                    loss_renoised_classification = loss_renoised_classification * mask
                loss_renoised_classification = loss_renoised_classification.mean()
                loss += loss_renoised_classification * self.renoised_classification_loss_scale
                loss_dict.update({f"{prefix}/renoised_classification_loss_{suffix}": loss_renoised_classification})
            loss_dict.update({f"{prefix}/student_loss_{suffix}": loss})
        elif self.phase == "disc":
            out_true = self.discriminate(disc, unet, x_true, t_true, emb, x_noisy)
            loss = F.relu(1 - out_true).sum(dim=1).mean() + F.relu(1 + out_false).sum(dim=1).mean()

            # logging
            bs = out_true.shape[0]
            acc_true = (out_true.view(bs, -1).mean(dim=1) > 0).sum().item() / bs
            acc_false = (out_false.view(bs, -1).mean(dim=1) < 0).sum().item() / bs
            mean_true = out_true.mean().item()
            mean_false = out_false.mean().item()
            loss_dict.update({f"disc_{prefix}/acc_true_{suffix}": acc_true})
            loss_dict.update({f"disc_{prefix}/acc_false_{suffix}": acc_false})
            loss_dict.update({f"disc_{prefix}/acc_total_{suffix}": (acc_true + acc_false) / 2})
            loss_dict.update({f"disc_{prefix}/mean_logit_true_{suffix}": mean_true})
            loss_dict.update({f"disc_{prefix}/mean_logit_false_{suffix}": mean_false})
        return loss, loss_dict

    def get_adversial_inputs(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        pred_noise: torch.Tensor,
        x_start: torch.Tensor,
        old_unet: AdjustedUNet,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.disc_input_mode in ["x0", "x0_no_ladd"]:
            x0_pred = self.predict_x0(x_noisy, t, pred_noise, clip_denoised=True)
            x_false = x0_pred
            t_false = torch.zeros_like(t)
        elif self.disc_input_mode == "x_t-1":
            # x_false = self.sample_xt_1(x_noisy, t, clip_denoised=True)
            x0_pred = self.predict_x0(x_noisy, t, pred_noise, clip_denoised=True)
            t_false = torch.relu(t - 1)
            noise = torch.randn_like(x0_pred)
            x_false = self.q_sample(x_start=x0_pred, t=t_false, noise=noise)
        elif self.disc_input_mode in ["x0_renoised", "x0_renoised_no_ladd"]:
            x0_pred = self.predict_x0(x_noisy, t, pred_noise, clip_denoised=True)
            t_false = self.sample_renoise_timestep(t.shape)
            noise = torch.randn_like(x0_pred)
            x_false = self.q_sample(x_start=x0_pred, t=t_false, noise=noise)

        if self.disc_input_mode == "x0":
            x_true = x_start
            t_true = torch.zeros_like(t)
        elif self.disc_input_mode == "x0_no_ladd":
            with torch.no_grad():
                old_noise_pred = old_unet.just_reconstruction(x_noisy, t)
                x_true = self.predict_x0(x_noisy, t, old_noise_pred, clip_denoised=True)
            t_true = torch.zeros_like(t)
        elif self.disc_input_mode == "x0_renoised_no_ladd":
            with torch.no_grad():
                old_noise_pred = old_unet.just_reconstruction(x_noisy, t)
                x_true = self.predict_x0(x_noisy, t, old_noise_pred, clip_denoised=True)
            t_true = self.sample_renoise_timestep(t.shape)
            noise = torch.randn_like(x_true)
            x_true = self.q_sample(x_start=x_true, t=t_true, noise=noise)
        else:
            t_true = self.sample_renoise_timestep(t.shape)
            noise = torch.randn_like(x_start)
            x_true = self.q_sample(x_start=x_start, t=t_true, noise=noise)
        return x_false, t_false, x_true, t_true, x0_pred

    def p_losses(self, x_start, t, noise=None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        prefix = "train" if self.training else "val"
        loss, loss_dict = super().p_losses(x_start, t, noise)
        old_unet: AdjustedUNet = self.old_model.model.diffusion_model
        new_unet = None if self.new_model is None else self.new_model.model.diffusion_model
        old_classes_mask = torch.any(self.batch_classes.unsqueeze(-1) == self.old_classes.to(self.device), dim=-1)
        new_classes_mask = torch.any(self.batch_classes.unsqueeze(-1) == self.new_classes.to(self.device), dim=-1)

        # adversarial diffusion distillation
        if self.use_old_and_new:
            assert new_unet is not None
            loss_old = torch.tensor(0)
            if old_classes_mask.sum() != 0:
                loss_old, loss_dict = self.get_adversarial_loss(
                    x_noisy=self.x_noisy[old_classes_mask],
                    t=t[old_classes_mask],
                    pred_noise=self.model_pred[old_classes_mask],
                    x_start=x_start[old_classes_mask],
                    classes=self.batch_classes[old_classes_mask],
                    repr_transformer=self.old_model.transform_representations,
                    unet=old_unet,
                    disc=self.old_disc,
                    classifier=self.old_model.classifier,
                    loss_dict=loss_dict,
                    prefix=prefix,
                    suffix="old",
                )
            loss_new = torch.tensor(0)
            if new_classes_mask.sum() != 0:
                loss_new, loss_dict = self.get_adversarial_loss(
                    x_noisy=self.x_noisy[new_classes_mask],
                    t=t[new_classes_mask],
                    pred_noise=self.model_pred[new_classes_mask],
                    x_start=x_start[new_classes_mask],
                    classes=self.batch_classes[new_classes_mask],
                    repr_transformer=self.new_model.transform_representations,
                    unet=new_unet,
                    disc=self.new_disc,
                    classifier=self.new_model.classifier,
                    loss_dict=loss_dict,
                    prefix=prefix,
                    suffix="new",
                )
            batch = new_classes_mask.sum() + old_classes_mask.sum()
            adv_loss = self.adv_loss_scale * (
                loss_new * 2 * (new_classes_mask.sum() / batch) * self.new_samples_weight
                + loss_old * 2 * (old_classes_mask.sum() / batch) * self.old_samples_weight
            )
            loss += adv_loss
            loss_dict.update({f"{prefix}/loss_diffusion_adv": adv_loss})
        else:
            loss_adv, loss_dict = self.get_adversarial_loss(
                x_noisy=self.x_noisy,
                t=t,
                pred_noise=self.model_pred,
                x_start=x_start,
                classes=self.batch_classes,
                repr_transformer=self.old_model.transform_representations,
                unet=old_unet,
                disc=self.old_disc,
                classifier=self.old_model.classifier,
                loss_dict=loss_dict,
                prefix=prefix,
                suffix="total",
            )
            loss += self.adv_loss_scale * loss_adv
            loss_dict.update({f"{prefix}/loss_diffusion_adv": loss_adv})
        return loss, loss_dict
