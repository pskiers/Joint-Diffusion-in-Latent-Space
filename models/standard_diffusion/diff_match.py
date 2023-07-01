import torch
import torch.nn as nn
from einops import rearrange
import kornia as K
from .ssl_joint_diffusion import SSLJointDiffusion


class DiffMatch(SSLJointDiffusion):
    def __init__(
            self,
            min_confidence=0.95,
            *args,
            **kwargs
        ):
        super().__init__(
            *args,
            **kwargs
        )
        self.min_confidence = min_confidence
        self.raw_imgs = None
        self.augmentation = K.augmentation.ImageSequential(
            K.augmentation.RandomAffine(degrees=0, translate=(0.125, 0.125)))
        self.strong_augmentation = K.augmentation.AugmentationSequential(
            K.augmentation.auto.RandAugment(n=2, m=10))
        if kwargs.get("ckpt_path", None) is not None:
            ignore_keys = kwargs.get("ignore_keys", [])
            only_model = kwargs.get("load_only_unet", False)
            self.init_from_ckpt(
                kwargs["ckpt_path"],
                ignore_keys=ignore_keys,
                only_model=only_model
            )

    def get_input(self, batch, k):
        if self.training:
            self.raw_imgs = batch[k]
            if len(self.raw_imgs.shape) == 3:
                self.raw_imgs = self.raw_imgs[..., None]
            self.raw_imgs = rearrange(self.raw_imgs, 'b h w c -> b c h w')
            self.raw_imgs = self.raw_imgs.to(
                memory_format=torch.contiguous_format).float()
            self.raw_imgs = self.raw_imgs.to(self.device)
        return super().get_input(batch, k)

    def p_losses(self, x_start, t, noise=None):
        loss, loss_dict = super().p_losses(x_start, t, noise)

        if not self.training:
            return loss, loss_dict
        prefix = 'train'

        with torch.no_grad():
            weakly_augmented = self.augmentation(self.raw_imgs).detach()
            weak_rep = self.model.diffusion_model.just_representations(
                weakly_augmented,
                torch.ones(weakly_augmented.shape[0], device=self.device),
                pooled=False
            )
            weak_rep = self.transform_representations(weak_rep)
            weak_preds = nn.functional.softmax(
                self.classifier(weak_rep), dim=1).detach()
            pseudo_labels = weak_preds.argmax(dim=1)
            above_threshold_idx, = (
                weak_preds.max(dim=1).values > self.min_confidence
            ).nonzero(as_tuple=True)
            pseudo_labels = pseudo_labels[above_threshold_idx]

            loss_dict.update(
                {f'{prefix}/ssl_above_threshold': len(above_threshold_idx) / len(weak_preds)})
            loss_dict.update({f'{prefix}/ssl_max_confidence': weak_preds.max()})
            if len(above_threshold_idx) == 0:
                return loss, loss_dict

            strongly_augmented = self.strong_augmentation((self.raw_imgs[above_threshold_idx]))
            strongly_augmented = self.cutout(strongly_augmented, level=1).detach()

        strong_rep = self.model.diffusion_model.just_representations(
            strongly_augmented,
            torch.ones(strongly_augmented.shape[0], device=self.device),
            pooled=False
        )
        strong_rep = self.transform_representations(strong_rep)
        preds = self.classifier(strong_rep)
        ssl_loss = nn.functional.cross_entropy(preds, pseudo_labels)

        loss += ssl_loss * len(preds) / len(weak_preds)
        loss_dict.update({f'{prefix}/loss_ssl_classification': ssl_loss})
        loss_dict.update({f'{prefix}/loss': loss})
        accuracy = torch.sum(
            torch.argmax(preds, dim=1) == pseudo_labels) / len(pseudo_labels)
        loss_dict.update({f'{prefix}/ssl_accuracy': accuracy})

        return loss, loss_dict

    def cutout(self, img_batch, level, fill=0.5):

        """
        Apply cutout to torch tensor of shape (batch, height, width, channel) at the specified level.
        """
        size = 1 + int(level * min(img_batch.shape[1:3]) * 0.499)
        batch, img_height, img_width = img_batch.shape[0:3]
        height_loc = torch.randint(low=0, high=img_height, size=[batch])
        width_loc = torch.randint(low=0, high=img_width, size=[batch])
        x_uppers = (height_loc - size // 2)
        x_uppers *= (x_uppers >= 0)
        x_lowers = (height_loc + size // 2)
        x_lowers -= (x_lowers >= img_height) * (x_lowers - img_height - 1)
        y_uppers = (width_loc - size // 2)
        y_uppers *= (y_uppers >= 0)
        y_lowers = (width_loc + size // 2)
        y_lowers -= (y_lowers >= img_width) * (y_lowers - img_width - 1)

        for img, x_upper, x_lower, y_upper, y_lower in zip(img_batch, x_uppers, x_lowers, y_uppers, y_lowers):
            img[x_upper:x_lower, y_upper:y_lower] = fill
        return img_batch
