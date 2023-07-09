import torch
import torch.nn as nn
from einops import rearrange
import kornia as K
from .joint_diffusion import JointDiffusion
from ..representation_transformer import RepresentationTransformer


class SSLJointDiffusion(JointDiffusion):
    def __init__(
            self,
            classification_loss_scale=1.0,
            supervised_skip=0,
            *args,
            **kwargs
        ):
        super().__init__(
            *args,
            **kwargs
        )
        self.supervised_imgs = None
        self.supervised_labels = None
        self.supervised_dataloader = None
        self.supervised_iterator = None
        self.supervised_skip_n = supervised_skip
        self.supervised_skip_current = supervised_skip
        self.classification_loss_scale = classification_loss_scale

        img_size = self.image_size
        self.augmentation = K.augmentation.ImageSequential(
            K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.25),
            K.augmentation.RandomResizedCrop(
                (img_size, img_size), scale=(0.5, 1), p=0.25),
            K.augmentation.RandomRotation((-30, 30), p=0.25),
            # K.augmentation.RandomHorizontalFlip(0.5),
            K.augmentation.RandomContrast((0.6, 1.8), p=0.25),
            K.augmentation.RandomSharpness((0.4, 2), p=0.25),
            K.augmentation.RandomBrightness((0.6, 1.8), p=0.25),
            K.augmentation.RandomMixUpV2(p=0.5),
        )
        if kwargs.get("ckpt_path", None) is not None:
            ignore_keys = kwargs.get("ignore_keys", [])
            only_model = kwargs.get("load_only_unet", False)
            self.init_from_ckpt(
                kwargs["ckpt_path"], ignore_keys=ignore_keys, only_model=only_model)

    def get_input(self, batch, k):
        model_input = super().get_input(batch, k)
        if self.training is True:
            # delete labels for unsupervised data
            self.batch_classes = None
            self.x_start = None

            if self.supervised_skip_current > 0:
                self.supervised_skip_current -= 1
            else:
                try:
                    b = next(self.supervised_iterator)
                except (StopIteration, TypeError):
                    self.supervised_iterator = iter(self.supervised_dataloader)
                    b = next(self.supervised_iterator)
                sup_imgs = b[k]
                if len(sup_imgs.shape) == 3:
                    sup_imgs = sup_imgs[..., None]
                sup_imgs = rearrange(sup_imgs, 'b h w c -> b c h w')
                sup_imgs = sup_imgs.to(
                    memory_format=torch.contiguous_format).float().to(self.device)
                self.supervised_imgs = self.augmentation(sup_imgs).detach()
                self.supervised_labels = b[self.classification_key].to(self.device)
                self.supervised_skip_current = self.supervised_skip_n

        return model_input

    def p_losses(self, x_start, t, noise=None):
        loss, loss_dict = super().p_losses(x_start, t, noise)
        if self.supervised_imgs is not None:

            representations = self.model.diffusion_model.just_representations(
                self.supervised_imgs,
                torch.ones(self.supervised_imgs.shape[0], device=self.device),
                pooled=False
            )
            if isinstance(representations, list): # TODO refactor this shit
                representations = self.transform_representations(representations)
                preds = self.classifier(representations)
            else:
                preds = representations

            prefix = 'train' if self.training else 'val'

            loss_classification = nn.functional.cross_entropy(
                preds, self.supervised_labels) * self.classification_loss_scale
            loss += loss_classification
            loss_dict.update(
                {f'{prefix}/loss_classification': loss_classification})
            loss_dict.update({f'{prefix}/loss': loss})
            accuracy = torch.sum(torch.argmax(preds, dim=1) == self.supervised_labels) / len(self.supervised_labels)
            loss_dict.update({f'{prefix}/accuracy': accuracy})

            self.supervised_imgs = None
            self.supervised_labels = None
        return loss, loss_dict


class SSLJointDiffusionAttention(SSLJointDiffusion):
    def __init__(
            self,
            attention_config,
            *args,
            **kwargs
        ):
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

