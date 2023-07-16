import math
from typing import Optional
import torch
import torch.nn as nn
from ..standard_diffusion import DiffMatch
from ..wide_resnet import Wide_ResNet
from ldm.util import count_params
from ldm.modules.ema import LitEma
import kornia as K


class FixMatchEma(LitEma):
    def forward(self,model):
        decay = self.decay

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name


class FixMatch(DiffMatch):
    def __init__(self, min_confidence=0.95, *args, **kwargs):
        super().__init__(min_confidence, *args, **kwargs)
        self.model = Wide_ResNet(depth=28, num_classes=10, widen_factor=2, drop_rate=0)
        count_params(self.model, verbose=True)
        self.model_ema = FixMatchEma(self.model, decay=0.999)
        print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        fixmatch_policy = [
            [("auto_contrast", 0, 1)],
            [("brightness", 0.05, 0.95)],
            [("color", 0.05, 0.95)],
            [("contrast", 0.05, 0.95)],
            [("equalize", 0, 1)],
            [("posterize", 4, 8)],
            [("rotate", -30.0, 30.0)],
            [("sharpness", 0.05, 0.95)],
            [("shear_x", -0.3, 0.3)],
            [("shear_y", -0.3, 0.3)],
            [("solarize", 0.0, 1.0)],
            [("translate_x", -0.3, 0.3)],
            [("translate_x", -0.3, 0.3)],
        ]

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
            K.augmentation.auto.RandAugment(n=2, m=10, policy=fixmatch_policy),
            K.augmentation.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                     std=(0.2471, 0.2435, 0.2616)),
        )
        self.scheduler = None

    def configure_optimizers(self):
        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': 5e-4},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.SGD(
            grouped_parameters,
            lr=0.03,
            momentum=0.9,
            nesterov=True
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
            strongly_augmented = self.cutout(strongly_augmented, level=1).detach()

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

    def on_train_batch_end(self, *args, **kwargs):
        self.scheduler.step()
        return super().on_train_batch_end(*args, **kwargs)
