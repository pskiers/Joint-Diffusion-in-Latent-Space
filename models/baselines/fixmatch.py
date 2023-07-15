import math
from typing import Optional
import torch
import torch.nn as nn
from ..standard_diffusion import DiffMatch
from ..wide_resnet import Wide_ResNet
from ldm.util import count_params
from ldm.modules.ema import LitEma
import kornia as K


class FixMatch(DiffMatch):
    def __init__(self, min_confidence=0.95, *args, **kwargs):
        super().__init__(min_confidence, *args, **kwargs)
        self.model = Wide_ResNet(28, 2, 0, 10)
        count_params(self.model, verbose=True)
        self.model_ema = LitEma(self.model)
        print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.val_aug = K.augmentation.ImageSequential(
            K.augmentation.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                     std=(0.2471, 0.2435, 0.2616)),
        )
        self.labeled_aug = K.augmentation.ImageSequential(
            K.augmentation.RandomHorizontalFlip(),
            K.augmentation.RandomCrop((32, 32),
                                      padding=int(32*0.125),
                                      padding_mode='reflect'),
            K.augmentation.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                     std=(0.2471, 0.2435, 0.2616)),
        )
        self.augmentation = K.augmentation.ImageSequential(
            K.augmentation.RandomHorizontalFlip(),
            K.augmentation.RandomCrop((32, 32),
                                      padding=int(32*0.125),
                                      padding_mode='reflect'),
            K.augmentation.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                     std=(0.2471, 0.2435, 0.2616)),
        )
        self.strong_augmentation = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(),
            K.augmentation.RandomCrop((32, 32),
                                      padding=int(32*0.125),
                                      padding_mode='reflect'),
            K.augmentation.auto.RandAugment(n=2, m=10),
            K.augmentation.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                     std=(0.2471, 0.2435, 0.2616)),
        )
        self.scheduler = None

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.03,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4
        )

        def _lr_lambda(current_step):
            num_warmup_steps = 0
            num_training_steps = 2**20
            num_cycles = 7./16.

            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            no_progress = float(current_step - num_warmup_steps) / \
                float(max(1, num_training_steps - num_warmup_steps))
            return max(0., math.cos(math.pi * num_cycles * no_progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, -1)
        return optimizer

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        self.scheduler.step()
        return

    def p_losses(self, x_start, t, noise=None):
        loss, loss_dict = torch.zeros(1, device=self.device), {}
        if self.batch_classes is not None:
            prefix = 'train' if self.training else 'val'
            x = self.labeled_aug(x_start) if self.training else self.val_aug(x_start)
            self.batch_class_predictions = self.model(x)

            loss_classification = nn.functional.cross_entropy(
                self.batch_class_predictions, self.batch_classes)
            loss += loss_classification
            loss_dict.update(
                {f'{prefix}/loss_classification': loss_classification})
            loss_dict.update({f'{prefix}/loss': loss})
            accuracy = torch.sum(torch.argmax(
                self.batch_class_predictions, dim=1) == self.batch_classes) / len(self.batch_classes)
            loss_dict.update({f'{prefix}/accuracy': accuracy})
        if self.supervised_imgs is not None:

            prefix = 'train' if self.training else 'val'
            x = self.labeled_aug(self.supervised_imgs) if self.training else self.val_aug(self.supervised_imgs)
            preds = self.model(x)

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

        if not self.training:
            return loss, loss_dict
        prefix = 'train'

        with torch.no_grad():
            weakly_augmented = self.augmentation(self.raw_imgs).detach()
            weak_preds = self.model(
                weakly_augmented
            )

            weak_preds = nn.functional.softmax(weak_preds, dim=1).detach()
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

            strongly_augmented = self.strong_augmentation((self.raw_imgs[above_threshold_idx])).detach()

        preds = self.model(
            strongly_augmented
        )
        ssl_loss = nn.functional.cross_entropy(preds, pseudo_labels)

        loss += ssl_loss * len(preds) / len(weak_preds)
        loss_dict.update({f'{prefix}/loss_ssl_classification': ssl_loss})
        loss_dict.update({f'{prefix}/loss': loss})
        accuracy = torch.sum(
            torch.argmax(preds, dim=1) == pseudo_labels) / len(pseudo_labels)
        loss_dict.update({f'{prefix}/ssl_accuracy': accuracy})

        return loss, loss_dict
