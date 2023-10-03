import torch
import torch.nn as nn
import kornia as K
import kornia.augmentation as aug
from ldm.models.diffusion.ddpm import DDPM
from ldm.util import default
from ..representation_transformer import RepresentationTransformer


class JointDiffusionNoisyClassifier(DDPM):
    def __init__(self,
                 unet_config,
                 classifier_in_features,
                 classifier_hidden,
                 num_classes,
                 dropout=0,
                 sample_grad_scale=60,
                 classification_key=1,
                 first_stage_key="image",
                 conditioning_key=None,
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
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_features, classifier_hidden),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            # nn.ReLU(),
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

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

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

    def log_images(self,
                   batch,
                   N=8,
                   n_row=4,
                   sample=True,
                   return_keys=None,
                   sample_classes=None,
                   **kwargs):
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
        if self.gradient_guided_sampling is True:
            return self.grad_guided_p_mean_variance(x, t, clip_denoised)
        else:
            model_out = self.apply_model(x, t)
            if self.parameterization == "eps":
                x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
            elif self.parameterization == "x0":
                x_recon = model_out
            if clip_denoised:
                x_recon.clamp_(-1., 1.)

            model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
            return model_mean, posterior_variance, posterior_log_variance

    def grad_guided_p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.guided_apply_model(x, t)

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
        if not hasattr(self.model.diffusion_model, 'forward_input_blocks'): # TODO refactor this shit
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_start = None
        self.gradient_guided_sampling = False

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
                torch.ones(self.x_start.shape[0], device=self.device),
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


class JointDiffusionAugmentations(JointDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        if kwargs.get("ckpt_path", None) is not None:
            ignore_keys = kwargs.get("ignore_keys", [])
            only_model = kwargs.get("load_only_unet", False)
            self.init_from_ckpt(kwargs["ckpt_path"], ignore_keys=ignore_keys, only_model=only_model)

    def get_input(self, batch, k):
        out = super().get_input(batch, k)
        self.x_start = self.augmentation(out)
        return out


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
        if kwargs.get("ckpt_path", None) is not None:
            ignore_keys = kwargs.get("ignore_keys", [])
            only_model = kwargs.get("load_only_unet", False)
            self.init_from_ckpt(kwargs["ckpt_path"], ignore_keys=ignore_keys, only_model=only_model)

    def transform_representations(self, representations):
        return representations
