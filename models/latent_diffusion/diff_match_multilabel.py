import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from .joint_latent_diffusion_multilabel import JointLatentDiffusionMultilabel
from ..representation_transformer import RepresentationTransformer
from ..adjusted_unet import AdjustedUNet


class LatentDiffMatchPoolingMultilabel(JointLatentDiffusionMultilabel):
    def __init__(
        self,
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
        classify_every_n=1,
        classify_every_n_times=1,
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
        **kwargs
    ):
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
        self.classify_every_n = classify_every_n
        self.classify_every_n_times = classify_every_n_times
        self.classification_counter = 0
        self.classification_times_counter = 0

    def get_sampl(self):
        print(
            "sampling_method, gradient_guided_samplings",
            self.sampling_method,
            self.gradient_guided_sampling,
        )

    def get_input(
        self,
        batch,
        k,
        return_first_stage_outputs=False,
        force_c_encode=False,
        cond_key=None,
        return_original_cond=False,
        bs=None,
    ):
        if self.training is True:
            return super().get_input(
                batch[0],
                k,
                return_first_stage_outputs,
                force_c_encode,
                cond_key,
                return_original_cond,
                bs,
            )
        return super().get_input(
            batch,
            k,
            return_first_stage_outputs,
            force_c_encode,
            cond_key,
            return_original_cond,
            bs,
        )

    def get_train_classification_input(self, batch, k):
        x = batch[0][k]
        x = self.to_latent(x, arrange=True)

        y = batch[0][self.classification_key]

        # _, weak_img, strong_img = batch[1][0]
        # weak_img = self.to_latent(weak_img, arrange=True)
        # strong_img = self.to_latent(strong_img, arrange=True)

        return x, y  # , weak_img, strong_img

    def get_valid_classification_input(self, batch, k):
        x = batch[k]
        x = self.to_latent(x, arrange=True)
        y = batch[self.classification_key]
        return x, y

    def train_classification_step(self, batch, loss):
        if self.classification_start > 0:
            self.classification_start -= 1
            return loss
        if self.classification_counter % self.classify_every_n != 0 and self.classify_every_n != 1:
            self.classification_counter += 1
            return loss
        self.classification_times_counter += 1
        if self.classification_times_counter % self.classify_every_n_times == 0:
            self.classification_counter += 1

        loss_dict = {}
        # x, y, weak_img, strong_img = self.get_train_classification_input(
        #     batch, self.first_stage_key)
        # t = torch.zeros((x.shape[0]*(2*self.mu+1),), device=self.device).long()

        # inputs = interleave(
        #     torch.cat((x, weak_img, strong_img)), 2*self.mu+1)

        x, y = self.get_train_classification_input(batch, self.first_stage_key)
        t = torch.zeros((x.shape[0]), device=self.device).long()

        inputs = x

        unet: AdjustedUNet = self.model.diffusion_model
        representations = unet.just_representations(inputs, t, pooled=False)
        representations = self.transform_representations(representations)
        logits = self.classifier(representations)

        # logits = de_interleave(logits, 2*self.mu+1)
        # preds_x = logits[:self.batch_size]
        # preds_weak, preds_strong = logits[self.batch_size:].chunk(2)
        # del logits

        preds_x = logits[: self.batch_size]

        loss_classification = nn.functional.binary_cross_entropy_with_logits(
            preds_x,
            y.float()[:, : self.num_classes],
            pos_weight=self.BCEweights.to(self.device),
            reduction="mean",
        )
        loss += loss_classification * self.classification_loss_weight
        loss_dict.update({"train/loss_classification": loss_classification})

        accuracy = accuracy_score(y[:, : self.num_classes].cpu(), preds_x.cpu() >= 0.5)
        loss_dict.update({"train/loss": loss})
        loss_dict.update({"train/accuracy": accuracy})

        if preds_x.shape[1] != y.shape[1]:  # means one class less in training
            self.auroc_train.update(preds_x, y[:, :-1].int())
        else:
            self.auroc_train.update(preds_x, y.int())

        self.log("train/auroc", self.auroc_train, on_step=False, on_epoch=True)

        # pseudo_label = torch.softmax(preds_weak.detach(), dim=-1)
        # max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        # mask = max_probs.ge(self.min_confidence).float()
        # ssl_loss = (nn.functional.cross_entropy(
        #     preds_strong, targets_u, reduction='none') * mask).mean()
        # loss += ssl_loss * self.classification_loss_weight
        # accuracy = torch.sum(
        #     (torch.argmax(preds_strong, dim=1) == targets_u) * mask
        # ) / mask.sum() if mask.sum() > 0 else 0
        # loss_dict.update(
        #     {'train/ssl_above_threshold': mask.mean().item()})
        # loss_dict.update({'train/ssl_max_confidence': mask.max().item()})
        # loss_dict.update({'train/loss_ssl_classification': ssl_loss})
        # loss_dict.update({'train/loss': loss})
        # loss_dict.update({'train/ssl_accuracy': accuracy})

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        return loss

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        if type(batch[0]) == list:
            batch = (batch[1][0], batch[1][1])
        return super().log_images(
            batch=batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs
        )


class LatentDiffMatchAttentionMultiLabel(LatentDiffMatchPoolingMultilabel):
    def __init__(
        self,
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
        **kwargs
    ):
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
