import torch
import torch.nn as nn
from ldm.models.diffusion.ddpm import LatentDiffusion


class JointLatentDiffusionNoisyClassifier(LatentDiffusion):
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

        self.batch_classes = None
        self.batch_class_predictions = None
        # Attributes that will store img labels and labels predictions
        # This is really ugly but since we are unable to change the parent classes and we don't want to copy-paste
        # code (especially that we'd have to copy a lot), this solution seems to be marginally better.

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
            raise NotImplementedError("This feature is not available for this model")

        x_recon, representations = self.model(x_noisy, t, **cond)
        representations = [torch.flatten(z_i, start_dim=1) for z_i in representations]
        representations = torch.concat(representations, dim=1)
        self.batch_class_predictions = self.classifier(representations)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def p_losses(self, x_start, cond, t, noise=None):
        loss, loss_dict = super().p_losses(x_start, cond, t, noise)

        prefix = 'train' if self.training else 'val'

        loss_classification = nn.functional.cross_entropy(self.batch_class_predictions, self.batch_classes)
        loss += loss_classification
        loss_dict.update({f'{prefix}/loss_classification': loss_classification})
        loss_dict.update({f'{prefix}/loss': loss})
        accuracy = (torch.argmax(self.batch_class_predictions) == self.batch_classes).sum() / len(self.batch_classes)
        loss_dict.update({f'{prefix}/accuracy': accuracy})

        return loss, loss_dict
