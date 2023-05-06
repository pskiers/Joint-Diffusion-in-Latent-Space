import torch
import torch.nn as nn
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.diffusionmodules.util import noise_like


class JointLatentDiffusionNoisyClassifier(LatentDiffusion):
    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        classifier_in_features,
        classifier_hidden,
        num_classes,
        dropout=0,
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
        self.num_classes = num_classes
        self.classification_key = classification_key
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_features, classifier_hidden),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(classifier_hidden, self.num_classes)
        )
        self.gradient_guided_sampling = True
        self.sample_grad_scale = sample_grad_scale
        self.batch_classes = None
        self.batch_class_predictions = None
        self.sample_classes = None
        # Attributes that will store img labels and labels predictions
        # This is really ugly but since we are unable to change the parent classes and we don't want to copy-paste
        # code (especially that we'd have to copy a lot), this solution seems to be marginally better.
        if kwargs.get("ckpt_path", None) is not None:
            ignore_keys = kwargs.get("ignore_keys", [])
            only_model = kwargs.get("load_only_unet", False)
            self.init_from_ckpt(kwargs["ckpt_path"], ignore_keys=ignore_keys, only_model=only_model)

    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False, cond_key=None, return_original_cond=False, bs=None):
        self.batch_classes = batch[self.classification_key]
        return super().get_input(batch, k, return_first_stage_outputs, force_c_encode, cond_key, return_original_cond, bs)

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        if hasattr(self, "split_input_params"):
            raise NotImplementedError(
                "This feature is not available for this model")

        x_recon, representations = self.model(x_noisy, t, **cond)
        representations = [torch.flatten(z_i, start_dim=1)
                           for z_i in representations]
        representations = torch.concat(representations, dim=1)
        self.batch_class_predictions = self.classifier(representations)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def p_losses(self, x_start, cond, t, noise=None):
        loss, loss_dict = super().p_losses(x_start, cond, t, noise)

        if self.batch_classes is not None:
            prefix = 'train' if self.training else 'val'

            loss_classification = nn.functional.cross_entropy(
                self.batch_class_predictions, self.batch_classes)
            loss += loss_classification
            loss_dict.update(
                {f'{prefix}/loss_classification': loss_classification})
            loss_dict.update({f'{prefix}/loss': loss})
            accuracy = torch.sum(torch.argmax(
                self.batch_class_predictions, dim=1) == self.batch_classes) / len(self.batch_classes)
            loss_dict.update({f'{prefix}/accuracy': accuracy})
        return loss, loss_dict

    def log_images(
            self,
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
            **kwargs
        ):
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

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        if self.gradient_guided_sampling is True:
            return self.grad_guided_p_mean_variance(x, c, t, clip_denoised, return_codebook_ids, quantize_denoised,
                                                    return_x0, score_corrector, corrector_kwargs)
        else:
            return super().p_mean_variance(x, c, t, clip_denoised, return_codebook_ids, quantize_denoised,
                                           return_x0, score_corrector, corrector_kwargs)

    def grad_guided_p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t

        emb = self.model.diffusion_model.get_timestep_embedding(x, t_in, None)

        with torch.enable_grad():
            representations = self.model.diffusion_model.forward_input_blocks(x, None, emb)
            for h in representations:
                h.retain_grad()
            pooled_representations = self.model.diffusion_model.pool_representations(representations)
            pooled_representations = [torch.flatten(z_i, start_dim=1) for z_i in pooled_representations]
            pooled_representations = torch.concat(pooled_representations, dim=1)
            class_predictions = self.classifier(pooled_representations)
            # loss = -torch.log(torch.gather(class_predictions, 1, self.sample_classes.unsqueeze(dim=1))).sum()
            loss = nn.functional.cross_entropy(class_predictions, self.sample_classes, reduction="sum")
            loss.backward()
            representations = [(h + self.sample_grad_scale * h.grad).detach() for h in representations]

        model_out = self.model.diffusion_model.forward_output_blocks(x, None, emb, representations)

        if isinstance(model_out, tuple) and not return_codebook_ids:
            model_out = model_out[0]

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

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
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance
