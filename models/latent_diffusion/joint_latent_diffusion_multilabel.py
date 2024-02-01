import torch
from .joint_latent_diffusion_noisy_classifier import JointLatentDiffusionNoisyClassifier, JointLatentDiffusionNoisyAttention
from ..representation_transformer import RepresentationTransformer
import torch.nn as nn
from ..adjusted_unet import AdjustedUNet
import numpy as np
from sklearn.metrics import accuracy_score
from torchmetrics import AUROC

class JointLatentDiffusionMultilabel(JointLatentDiffusionNoisyClassifier):
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 classifier_in_features,
                 classifier_hidden,
                 num_classes,
                 dropout=0,
                 classification_loss_weight=1.0,
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
        super().__init__(
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            classifier_in_features=classifier_in_features,
            classifier_hidden=classifier_hidden,
            num_classes=num_classes,
            dropout=dropout,
            classification_loss_weight=classification_loss_weight,
            sample_grad_scale=0,
            classification_key=classification_key,
            augmentations=augmentations,
            num_timesteps_cond=num_timesteps_cond,
            cond_stage_key=cond_stage_key,
            cond_stage_trainable=cond_stage_trainable,
            concat_mode=concat_mode,
            cond_stage_forward=cond_stage_forward,
            conditioning_key=conditioning_key,
            scale_factor=scale_factor,
            scale_by_std=scale_by_std,
            *args,
            **kwargs
        )
        # self.x_start = None
        self.gradient_guided_sampling = False
        self.auroc_train = AUROC(num_classes=num_classes-1)
        self.auroc_val = AUROC(num_classes=num_classes-1)
        
        # counts from https://www.mdpi.com/2075-4426/13/10/1426 ->parametrize it!!!
        self.BCEweights = torch.Tensor([39.4, 43.6, 47.7, 492.9, 20.1, 7.4, 18.4, 65.5, 8.7, 23.0, 32.1, 16.7, 77.4, 4.6, 0.9]).to(self.device)

    def do_classification(self, x, t, y):
        unet: AdjustedUNet = self.model.diffusion_model
        representations = unet.just_representations(x, t, pooled=False)
        representations = self.transform_representations(representations)
        y_pred = self.classifier(representations)

        loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y.float(), pos_weight=self.BCEweights)
        accuracy = accuracy_score(y.cpu(), y_pred.cpu()>=0.5)
        return loss, accuracy, y_pred


    def train_classification_step(self, batch, loss):
        loss_dict = {}

        x, y = self.get_train_classification_input(batch, self.first_stage_key)
        t = torch.zeros((x.shape[0],), device=self.device).long()

        loss_classification, accuracy, y_pred = self.do_classification(x, t, y)
        loss += loss_classification * self.classification_loss_weight

        self.auroc_train.update(y_pred[:,:-1], y[:,:-1])
        self.log('train/auroc', self.auroc_train, on_step=False, on_epoch=True)
        loss_dict.update({'train/loss_classification': loss_classification})
        loss_dict.update({'train/loss_full': loss})
        loss_dict.update({'train/accuracy': accuracy})
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)


        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = self.get_valid_classification_input(batch, self.first_stage_key)
        t = torch.zeros((x.shape[0],), device=self.device).long()
        x_diff, c = self.get_input(batch, self.first_stage_key)

        loss, loss_dict_no_ema = self(x_diff, c)
        loss_cls, accuracy, _ = self.do_classification(x, t, y)
        loss_dict_no_ema.update({'val/loss_classification': loss_cls})
        loss_dict_no_ema.update({'val/loss_full': loss + loss_cls})
        loss_dict_no_ema.update({'val/accuracy': accuracy})


        with self.ema_scope():
            loss, loss_dict_ema = self(x_diff, c)
            loss_cls, accuracy, y_pred = self.do_classification(x, t, y)

            self.auroc_val.update(y_pred[:,:-1], y[:,:-1])
            self.log('val/auroc_ema', self.auroc_val, on_step=False, on_epoch=True)

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

