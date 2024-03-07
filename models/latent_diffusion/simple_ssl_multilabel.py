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


class LatentSSLMultilabel(JointLatentDiffusionMultilabel):
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
        self.min_confidence = min_confidence
        self.mu = mu
        self.batch_size = batch_size
        self.classification_start = classification_start

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
        x = batch[0][k]
        x = self.to_latent(x, arrange=True)

        y = batch[0][self.classification_key]

        return x, y 

    def get_valid_classification_input(self, batch, k):
        x = batch[k]
        x = self.to_latent(x, arrange=True)
        y = batch[self.classification_key]
        return x, y

    def train_classification_step(self, batch, loss):
        if self.classification_start > 0:
            self.classification_start -= 1
            return loss

        loss_dict = {}

        x, y = self.get_train_classification_input(
            batch, self.first_stage_key)
        t = torch.zeros((x.shape[0]), device=self.device).long()

        inputs = x

        unet: AdjustedUNet = self.model.diffusion_model
        representations = unet.just_representations(inputs, t, pooled=False)
        representations = self.transform_representations(representations)
        logits = self.classifier(representations)

        preds_x = logits[:self.batch_size]
        
        loss_classification = nn.functional.binary_cross_entropy_with_logits(preds_x, y.float()[:,:self.num_classes], 
                                                                             pos_weight=self.BCEweights.to(self.device), reduction="mean")
        loss += loss_classification * self.classification_loss_weight
        loss_dict.update(
            {'train/loss_classification': loss_classification})
        
        accuracy = accuracy_score(y[:,:self.num_classes].cpu(), preds_x.cpu()>=0.5)
        loss_dict.update({'train/loss': loss})
        loss_dict.update({'train/accuracy': accuracy})

        if preds_x.shape[1]!=y.shape[1]: #means one class less in training
            self.auroc_train.update(preds_x, y[:,:-1])
        else:
            self.auroc_train.update(preds_x[:,:-1], y[:,:-1])

        self.log('train/auroc', self.auroc_train, on_step=False, on_epoch=True)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        return loss
    
    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        if type(batch[0])==list:
            batch = (batch[1][0][0], batch[1][1])
        return super().log_images(batch=batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs)

class LatentSSLAttentionMultiLabel(LatentSSLMultilabel):
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
