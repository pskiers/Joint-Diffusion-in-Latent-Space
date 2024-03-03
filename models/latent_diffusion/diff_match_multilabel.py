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


class LatentDiffMatchPoolingMultilabel(JointLatentDiffusionMultilabel):
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 classifier_in_features,
                 classifier_hidden,
                 num_classes,
                 min_confidences,
                 mu=7,
                 batch_size=64,
                 classification_start=0,
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
                 weights=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

                 *args,
                 **kwargs):
        super().__init__(
            first_stage_config,
            cond_stage_config,
            classifier_in_features,
            classifier_hidden,
            num_classes,
            dropout,
            classification_loss_weight,
            classification_key,
            augmentations,
            num_timesteps_cond,
            cond_stage_key,
            cond_stage_trainable,
            concat_mode,
            cond_stage_forward,
            conditioning_key,
            scale_factor,
            scale_by_std,
            weights,
            *args,
            **kwargs
        )
        self.min_confidences = torch.tensor(min_confidences)
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

        _, weak_img, strong_img = batch[1][0]
        weak_img = self.to_latent(weak_img, arrange=True)
        strong_img = self.to_latent(strong_img, arrange=True)

        return x, y, weak_img, strong_img

    def get_valid_classification_input(self, batch, k):
        x = batch[k]
        x = self.to_latent(x, arrange=True)
        y = batch[self.classification_key]
        return x, y

    def train_classification_step(self, batch, loss):
        if self.classification_start > 0:
            self.classification_start -= 0
            return loss

        loss_dict = {}
        x, y, weak_img, strong_img = self.get_train_classification_input(
            batch, self.first_stage_key)
        t = torch.zeros((x.shape[0]*(2*self.mu+1),), device=self.device).long()

        inputs = interleave(
            torch.cat((x, weak_img, strong_img)), 2*self.mu+1)

        unet: AdjustedUNet = self.model.diffusion_model
        representations = unet.just_representations(inputs, t, pooled=False)
        representations = self.transform_representations(representations)
        logits = self.classifier(representations)

        logits = de_interleave(logits, 2*self.mu+1)
        preds_x = logits[:self.batch_size]
        preds_weak, preds_strong = logits[self.batch_size:].chunk(2)
        del logits

        loss_classification = nn.functional.binary_cross_entropy_with_logits(preds_x, y.float()[:, :self.num_classes],
                                                                             pos_weight=self.BCEweights.to(self.device), reduction="mean")
        loss += loss_classification * self.classification_loss_weight
        loss_dict.update(
            {'train/loss_classification': loss_classification})

        accuracy = torch.sum(
            (preds_x.detach() >= 0).long() == y[:, :self.num_classes]
        ) / torch.numel(y[:, :self.num_classes])

        loss_dict.update({'train/loss': loss})
        loss_dict.update({'train/accuracy': accuracy})

        if preds_x.shape[1] != y.shape[1]:  # means one class less in training
            self.auroc_train.update(preds_x, y[:, :-1])
        else:
            self.auroc_train.update(preds_x[:, :-1], y[:, :-1])

        self.log('train/auroc', self.auroc_train, on_step=False, on_epoch=True)

        pseudo_label = torch.sigmoid(preds_weak.detach())
        mask = torch.logical_or(
            pseudo_label >= self.min_confidences.to(self.device).unsqueeze(0),
            pseudo_label <= (-self.min_confidences.to(self.device).unsqueeze(0) + 1)
        ).float()
        targets_u = (pseudo_label >= 0.5).float()
        ssl_loss = (nn.functional.binary_cross_entropy_with_logits(
            preds_strong, targets_u, reduction='none') * mask).mean()
        loss += ssl_loss * self.classification_loss_weight
        accuracy = torch.sum(
            ((torch.sigmoid(preds_strong) >= 0.5).long() == targets_u.long()) * mask
        ) / mask.sum() if mask.sum() > 0 else 0
        loss_dict.update({'train/ssl_above_threshold': mask.mean().item()})
        for i in range(mask.shape[1]):
            loss_dict.update({f'train/ssl_class_{i}_above_threshold': mask[:, i].mean().item()})
        # loss_dict.update({'train/ssl_max_confidence': mask.max().item()})
        # for i in range(mask.shape[1]):
        #     loss_dict.update({f'train/ssl_class_{i}_max_confidence': mask[:, i].max().item()})
        loss_dict.update({'train/loss_ssl_classification': ssl_loss})
        loss_dict.update({'train/loss': loss})
        loss_dict.update({'train/ssl_accuracy': accuracy})
        for i in range(mask.shape[1]):
            accuracy = torch.sum(
                ((torch.sigmoid(preds_strong[:, i].detach()) >= 0.5).long() == targets_u[:, i].long()) * mask[:, i]
            ) / mask[:, i].sum() if mask[:, i].sum() > 0 else 0
            loss_dict.update({f'train/ssl_class_{i}_accuracy': accuracy})
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        if type(batch[0]) == list:
            batch = (batch[1][0][0], batch[1][1])
        return super().log_images(batch=batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs)


class LatentDiffMatchAttentionMultiLabel(LatentDiffMatchPoolingMultilabel):
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 attention_config,
                 min_confidence,
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
        self.min_confidence = torch.tensor(min_confidence)
        self.mu = mu
        self.batch_size = batch_size
        self.classification_start = classification_start
        self.classifier = RepresentationTransformer(**attention_config)

        self.init_ema()
        self.init_ckpt(**kwargs)

    def transform_representations(self, representations):
        return representations
