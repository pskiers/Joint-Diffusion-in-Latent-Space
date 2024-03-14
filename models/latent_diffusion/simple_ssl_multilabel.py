import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange
import kornia as K
from .ssl_joint_diffusion import SSLJointLatentDiffusion, SSLJointLatentDiffusionV2
from .joint_latent_diffusion import JointLatentDiffusion, JointLatentDiffusionAttention
from .joint_latent_diffusion_multilabel import JointLatentDiffusionMultilabel
from ..representation_transformer import RepresentationTransformer
from ..adjusted_unet import AdjustedUNet
from ..ddim import DDIMSamplerGradGuided
from ..utils import FixMatchEma, interleave, de_interleave
from sklearn.metrics import accuracy_score


class LatentSSLPoolingMultilabel(JointLatentDiffusionMultilabel):
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 classifier_in_features,
                 classifier_hidden,
                 num_classes,
                 classification_start=0,
                 dropout=0,
                 classification_loss_weight=1.0,
                 classification_key=1,
                 augmentations = True,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1,
                 scale_by_std=False,
                 weights=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

                 *args,
                 **kwargs):
        super().__init__(
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            classifier_in_features=classifier_in_features,
            classifier_hidden=classifier_hidden,
            num_classes=num_classes,
            dropout=dropout,
            classification_start=classification_start,
            classification_loss_weight=classification_loss_weight,
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
            weights=weights,
            *args,
            **kwargs
        )

    def get_sampl(self):
        print("sampling_method, gradient_guided_samplings", self.sampling_method, self.gradient_guided_sampling)
        
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
        if type(batch[0])==list:
            x = batch[0][k]
        else:
            x = batch[k]
        x = self.to_latent(x, arrange=True)
        y = batch[self.classification_key]
        return x, y 

    def get_valid_classification_input(self, batch, k):
        x = batch[k]
        x = self.to_latent(x, arrange=True)
        y = batch[self.classification_key]
        return x, y
    
    def get_input(self,
                  batch,
                  k,
                  return_first_stage_outputs=False,
                  force_c_encode=False,
                  cond_key=None,
                  return_original_cond=False,
                  bs=None):
        if type(batch[0])==list:
            batch = batch[0]
        # k=0 should mean img for tuple (img, label). 
        #Here it means sth different to have mathcing idx: (img, img_weak, img_strong)
        return super().get_input(
            batch,
            k,
            return_first_stage_outputs,
            force_c_encode,
            cond_key,
            return_original_cond,
            bs
        )

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch[1])
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        loss = self.train_classification_step(batch[0], loss)
        return loss

    def train_classification_step(self, batch, loss):
        if self.classification_start > 0:
            self.classification_start -= 1
            return loss
        
        if self.global_step%4!=0:
            return loss
        
        loss_dict = {}

        x, y = self.get_train_classification_input(batch, self.first_stage_key)
        t = torch.zeros((x.shape[0],), device=self.device).long()
            
        loss_classification, accuracy, y_pred = self.do_classification(x, t, y)
        self.auroc_train.update(y_pred[:,:self.used_n_classes], y[:,:self.used_n_classes])

        loss += loss_classification * self.classification_loss_weight

        self.log('train/auroc', self.auroc_train, on_step=False, on_epoch=True)
        loss_dict.update({'train/loss_classification': loss_classification})
        loss_dict.update({'train/loss_full': loss})
        loss_dict.update({'train/accuracy': accuracy})
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        return loss
    
    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        return super().log_images(batch=batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs)

class LatentSSLAttentionMultiLabel(LatentSSLPoolingMultilabel):
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
