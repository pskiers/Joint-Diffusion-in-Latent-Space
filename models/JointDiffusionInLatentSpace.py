import torch
import torch.nn as nn
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import default

class JointDiffusionInLatentSpace(LatentDiffusion):
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
            nn.ReLU(),
            nn.Linear(classifier_hidden, self.num_classes)
        )

    def shared_step(self, batch, **kwargs):
        x, c, classes = self.get_input(batch, self.first_stage_key)
        loss = self(x, c, classes)
        return loss

    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False, cond_key=None, return_original_cond=False, bs=None):
        ldm_input = super().get_input(batch, k, return_first_stage_outputs, force_c_encode, cond_key, return_original_cond, bs)
        classes = batch[self.classification_key]
        return *ldm_input, classes

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
            raise NotImplementedError("This feature is not available for this model")

        x_recon, representations = self.model(x_noisy, t, **cond)
        representations = [torch.flatten(z_i, start_dim=1) for z_i in representations]
        representations = torch.concat(representations, dim=1)
        predictions = self.classifier(representations)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0], predictions
        else:
            return x_recon, predictions

    def p_losses(self, x_start, cond, t, classes, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output, predictions = self.apply_model(x_noisy, t, cond)

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

        loss_classification = nn.functional.cross_entropy(predictions, classes)
        loss += loss_classification
        loss_dict.update({f'{prefix}/loss_classification': loss_classification})
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

