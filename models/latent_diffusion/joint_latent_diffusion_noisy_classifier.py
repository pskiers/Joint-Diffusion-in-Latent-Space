import torch
import torch.nn as nn
from einops import rearrange
import kornia as K
import kornia.augmentation as aug
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.ema import LitEma
from contextlib import contextmanager
from ..ddim import DDIMSamplerGradGuided
from ..representation_transformer import RepresentationTransformer
from ..adjusted_unet import AdjustedUNet


class JointLatentDiffusionNoisyClassifier(LatentDiffusion):
    def _init(self,
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
              **kwargs):
        kwargs.pop("ckpt_path", None)
        kwargs.pop("ignore_keys", [])
        #kwargs.update({"use_ema": False})
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

    def init_ema(self) -> None:
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    def init_ckpt(self, **kwargs):
        if kwargs.get("ckpt_path", None) is not None:
            ignore_keys = kwargs.get("ignore_keys", [])
            only_model = kwargs.get("load_only_unet", False)
            self.init_from_ckpt(kwargs["ckpt_path"], ignore_keys=ignore_keys, only_model=only_model)

    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 classifier_in_features,
                 classifier_hidden,
                 num_classes,
                 dropout=0,
                 classification_loss_weight=1.0,
                 sample_grad_scale=60,
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
        self.num_classes = num_classes
        self.classification_key = classification_key
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_features, classifier_in_features//8),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=dropout),
                nn.Linear(classifier_in_features//8, self.num_classes)
        )

        self.gradient_guided_sampling = True
        self.classification_loss_weight = classification_loss_weight
        self.sample_grad_scale = sample_grad_scale
        self.augmentations = None
        if augmentations is True:
            img_size = self.image_size
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

        self.init_ema()
        self.init_ckpt(**kwargs)

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

    @torch.no_grad()
    def to_latent(self, imgs, arrange=True):
        if len(imgs.shape) == 3:
            imgs = imgs[..., None]
        if arrange is True:
            imgs = rearrange(imgs, 'b h w c -> b c h w')
        imgs = imgs.to(
            self.device, memory_format=torch.contiguous_format).float()
        encoder_posterior = self.encode_first_stage(imgs)
        return self.get_first_stage_encoding(encoder_posterior).detach()

    @torch.no_grad()
    def get_train_classification_input(self, batch, k):
        x = batch[k]
        if self.augmentations is not None:
            x = self.augmentation(x)
        x = self.to_latent(x)
        y = batch[self.classification_key]
        return x, y

    @torch.no_grad()
    def get_valid_classification_input(self, batch, k):
        x = batch[k]
        x = self.to_latent(x)
        y = batch[self.classification_key]
        return x, y

    def do_classification(self, x, t, y):
        unet: AdjustedUNet = self.model.diffusion_model
        representations = unet.just_representations(x, t, pooled=False)
        representations = self.transform_representations(representations)
        y_pred = self.classifier(representations)

        loss = nn.functional.cross_entropy(y_pred, y)
        accuracy = torch.sum(torch.argmax(y_pred, dim=1) == y) / len(y)
        return loss, accuracy

    def train_classification_step(self, batch, loss):
        loss_dict = {}
        x, y = self.get_train_classification_input(batch, self.first_stage_key)

        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)

        loss_classification, accuracy = self.do_classification(x_noisy, t, y)
        loss += loss_classification * self.classification_loss_weight

        loss_dict.update({'train/loss_classification': loss_classification})
        loss_dict.update({'train/loss_full': loss})
        loss_dict.update({'train/accuracy': accuracy})

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        loss = self.train_classification_step(batch, loss)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = self.get_valid_classification_input(batch, self.first_stage_key)
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)

        loss, loss_dict_no_ema = self.shared_step(batch)
        loss_cls, accuracy = self.do_classification(x_noisy, t, y)
        loss_dict_no_ema.update({'val/loss_classification': loss_cls})
        loss_dict_no_ema.update({'val/loss_full': loss + loss_cls})
        loss_dict_no_ema.update({'val/accuracy': accuracy})

        with self.ema_scope():
            loss, loss_dict_ema = self.shared_step(batch)
            loss_cls, accuracy = self.do_classification(x_noisy, t, y)
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

    def transform_representations(self, representations):
        unet: AdjustedUNet = self.model.diffusion_model
        representations = unet.pool_representations(representations)
        representations = [torch.flatten(z_i, start_dim=1)
                           for z_i in representations]
        representations = torch.concat(representations, dim=1)
        return representations

    # Changed for for compatibility reason
    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if hasattr(self, "split_input_params"):
            raise NotImplementedError(
                "This feature is not available for this model")

        unet: AdjustedUNet = self.model.diffusion_model
        x_recon = unet.just_reconstruction(x_noisy, t)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def log_images(self,
                   batch,
                   N=8,
                   n_row=4,
                   sample=True,
                   ddim_steps=200,
                   ddim_eta=1,
                   return_keys=None,
                   quantize_denoised=True,
                   inpaint=True,
                   plot_denoise_rows=False,
                   plot_progressive_rows=True,
                   plot_diffusion_rows=True,
                   sample_classes=None,
                   **kwargs):
        self.sample_classes = torch.tensor(sample_classes).to(self.device)
        return super().log_images(
            batch,
            N,
            n_row,
            sample,
            ddim_steps,
            ddim_eta,
            return_keys,
            quantize_denoised,
            inpaint,
            plot_denoise_rows,
            plot_progressive_rows,
            plot_diffusion_rows,
            **kwargs
        )

    def p_mean_variance(self,
                        x,
                        c,
                        t,
                        clip_denoised: bool,
                        return_codebook_ids=False,
                        quantize_denoised=False,
                        return_x0=False,
                        score_corrector=None,
                        corrector_kwargs=None):
        if self.gradient_guided_sampling is True:
            return self.grad_guided_p_mean_variance(
                x=x,
                c=c,
                t=t,
                clip_denoised=clip_denoised,
                return_codebook_ids=return_codebook_ids,
                quantize_denoised=quantize_denoised,
                return_x0=return_x0,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs
            )
        else:
            return super().p_mean_variance(
                x,
                c,
                t,
                clip_denoised,
                return_codebook_ids,
                quantize_denoised,
                return_x0,
                score_corrector,
                corrector_kwargs
            )

    @torch.no_grad()
    def guided_apply_model(self, x, t):
        t_in = t
        unet: AdjustedUNet = self.model.diffusion_model

        emb = unet.get_timestep_embedding(x, t_in, None)

        with torch.enable_grad():
            representations = unet.forward_input_blocks(x, None, emb)
            for h in representations:
                h.retain_grad()
            pooled_representations = self.transform_representations(
                representations)
            class_predictions = self.classifier(pooled_representations)
            # loss = -torch.log(
            #     torch.gather(
            #         class_predictions,
            #         1,
            #         self.sample_classes.unsqueeze(dim=1)
            #     )
            # ).sum()
            loss = nn.functional.cross_entropy(
                class_predictions, self.sample_classes, reduction="sum")
            loss.backward()
            representations = [
                (h + self.sample_grad_scale * h.grad).detach()
                for h in representations
            ]

        model_out = unet.forward_output_blocks(x, None, emb, representations)
        return model_out

    def grad_guided_p_mean_variance(self,
                                    x,
                                    c,
                                    t,
                                    clip_denoised: bool,
                                    return_codebook_ids=False,
                                    quantize_denoised=False,
                                    return_x0=False,
                                    score_corrector=None,
                                    corrector_kwargs=None):
        model_out = self.guided_apply_model(x, t)

        if isinstance(model_out, tuple) and not return_codebook_ids:
            model_out = model_out[0]

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(
                self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(
                x_recon)
        (
            model_mean,
            posterior_variance,
            posterior_log_variance
        ) = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return (model_mean,
                    posterior_variance,
                    posterior_log_variance,
                    logits)
        elif return_x0:
            return (model_mean,
                    posterior_variance,
                    posterior_log_variance,
                    x_recon)
        else:
            return model_mean, posterior_variance, posterior_log_variance


class JointLatentDiffusionNoisyAttention(JointLatentDiffusionNoisyClassifier):
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 attention_config,
                 augmentations=True,
                 classification_loss_weight=1.0,
                 sample_grad_scale=60,
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
        self.gradient_guided_sampling = True
        self.sample_grad_scale = sample_grad_scale
        self.augmentations = None
        if augmentations is True:
            img_size = self.image_size
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

        self.classifier = RepresentationTransformer(**attention_config)

        self.init_ema()
        self.init_ckpt(**kwargs)

    def transform_representations(self, representations):
        return representations

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        if ddim:
            ddim_sampler = DDIMSamplerGradGuided(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                **kwargs
            )
        else:
            samples, intermediates = self.sample(
                cond=cond,
                batch_size=batch_size,
                return_intermediates=True,
                **kwargs
            )
        return samples, intermediates
