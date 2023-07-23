import math
from typing import Optional
from contextlib import contextmanager
import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
import kornia as K
from ..standard_diffusion import DiffMatch
from ..wide_resnet import Wide_ResNet
from ldm.util import count_params
from ldm.modules.ema import LitEma
from copy import deepcopy


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


class ModelEMA(object):
    def __init__(self, model, decay, device):
        self.ema = deepcopy(model)
        self.ema.to(device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])

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
                    assert key not in self.m_name2s_name


class FixMatch(pl.LightningModule):
    def __init__(self,
                 min_confidence=0.95,
                 mu=7,
                 batch_size=64,
                 img_key=0,
                 label_key=1,
                 unsup_img_key=0,
                 monitor="val/loss_ema",
                 ckpt_path=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_confidence = min_confidence
        self.mu = mu
        self.batch_size = batch_size
        self.model = Wide_ResNet(depth=28, num_classes=10, widen_factor=2, drop_rate=0)
        count_params(self.model, verbose=True)
        self.model_ema = FixMatchEma(self.model, decay=0.999)
        # self.model_ema = ModelEMA(self.model, 0.999, torch.device("cuda"))
        self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
        self.img_key = img_key
        self.label_key = label_key
        self.unsup_img_key = unsup_img_key
        self.use_ema = True

        # fixmatch_policy = [
        #     [("auto_contrast", 0, 1)],
        #     [("brightness", 0.05, 0.95)],
        #     [("color", 0.05, 0.95)],
        #     [("contrast", 0.05, 0.95)],
        #     [("equalize", 0, 1)],
        #     [("posterize", 4, 8)],
        #     [("rotate", -30.0, 30.0)],
        #     [("sharpness", 0.05, 0.95)],
        #     [("shear_x", -0.3, 0.3)],
        #     [("shear_y", -0.3, 0.3)],
        #     [("solarize", 0.0, 1.0)],
        #     [("translate_x", -0.3, 0.3)],
        #     [("translate_x", -0.3, 0.3)],
        # ]

        # self.val_aug = K.augmentation.ImageSequential(
        #     K.augmentation.Normalize(mean=(0.4914, 0.4822, 0.4465),
        #                              std=(0.2471, 0.2435, 0.2616)),
        # )
        # self.labeled_aug = K.augmentation.ImageSequential(
        #     K.augmentation.RandomHorizontalFlip(),
        #     K.augmentation.RandomCrop((32, 32),
        #                               padding=int(32*0.125),
        #                               padding_mode='reflect'),
        #     K.augmentation.Normalize(mean=(0.4914, 0.4822, 0.4465),
        #                              std=(0.2471, 0.2435, 0.2616)),
        # )
        # self.augmentation = K.augmentation.ImageSequential(
        #     K.augmentation.RandomHorizontalFlip(),
        #     K.augmentation.RandomCrop((32, 32),
        #                               padding=int(32*0.125),
        #                               padding_mode='reflect'),
        #     K.augmentation.Normalize(mean=(0.4914, 0.4822, 0.4465),
        #                              std=(0.2471, 0.2435, 0.2616)),
        # )
        # self.strong_augmentation = K.augmentation.AugmentationSequential(
        #     K.augmentation.RandomHorizontalFlip(),
        #     K.augmentation.RandomCrop((32, 32),
        #                               padding=int(32*0.125),
        #                               padding_mode='reflect'),
        #     K.augmentation.auto.RandAugment(n=2, m=10, policy=fixmatch_policy),
        #     K.augmentation.Normalize(mean=(0.4914, 0.4822, 0.4465),
        #                              std=(0.2471, 0.2435, 0.2616)),
        # )
        self.scheduler = None

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

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

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def get_train_input(self, batch):
        x = batch[0][self.img_key]
        # if len(x.shape) == 3:
        #     x = x[..., None]
        # x = rearrange(x, 'b h w c -> b c h w')
        # x = x.to(memory_format=torch.contiguous_format).float()
        # x = self.labeled_aug(x)
        y = batch[0][self.label_key]

        # unsup_img = batch[1][self.unsup_img_key]
        # if len(unsup_img.shape) == 3:
        #     unsup_img = unsup_img[..., None]
        # unsup_img = rearrange(unsup_img, 'b h w c -> b c h w')
        # unsup_img = unsup_img.to(memory_format=torch.contiguous_format).float()
        # weak_img = self.augmentation(unsup_img)
        # strong_img = self.strong_augmentation(unsup_img)
        # strong_img = self.cutout(strong_img, 1)

        weak_img, strong_img = batch[1][0]
        return x, y, weak_img, strong_img

    def get_val_input(self, batch):
        x = batch[self.img_key]
        # if len(x.shape) == 3:
            # x = x[..., None]
        # x = rearrange(x, 'b h w c -> b c h w')
        # x = x.to(memory_format=torch.contiguous_format).float()
        # x = self.val_aug(x)
        y = batch[self.label_key]
        return x, y

    # def shared_step(self, batch):
    #     x = self.get_input(batch, self.first_stage_key)
    #     loss, loss_dict = self(x)
    #     return loss, loss_dict

    def training_step(self, batch, batch_idx):
        x, y, weak_img, strong_img = self.get_train_input(batch)

        loss, loss_dict = torch.zeros(1, device=self.device), {}

        inputs = interleave(
            torch.cat((x, weak_img, strong_img)), 2*self.mu+1)
        logits = self.model(inputs)
        logits = de_interleave(logits, 2*self.mu+1)
        preds_x = logits[:self.batch_size]
        preds_weak, preds_strong = logits[self.batch_size:].chunk(2)
        del logits

        loss_classification = nn.functional.cross_entropy(preds_x, y, reduction="mean")
        loss += loss_classification
        accuracy = torch.sum(torch.argmax(preds_x, dim=1) == y) / len(y)
        loss_dict.update(
            {'train/loss_classification': loss_classification})
        loss_dict.update({'train/loss': loss})
        loss_dict.update({'train/accuracy': accuracy})

        pseudo_label = torch.softmax(preds_weak.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.min_confidence).float()
        ssl_loss = (nn.functional.cross_entropy(
            preds_strong, targets_u, reduction='none') * mask).mean()
        loss += ssl_loss
        accuracy = torch.sum(
            torch.argmax(preds_strong, dim=1) == targets_u * mask
        ) / mask.sum() if mask.sum() > 0 else 0
        loss_dict.update(
                {'train/ssl_above_threshold': mask.mean().item()})
        loss_dict.update({'train/ssl_max_confidence': mask.max().item()})
        loss_dict.update({'train/loss_ssl_classification': ssl_loss})
        loss_dict.update({'train/loss': loss})
        loss_dict.update({'train/ssl_accuracy': accuracy})
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = self.get_val_input(batch)

        loss_dict_no_ema = {}
        preds = self.model(x)
        loss = nn.functional.cross_entropy(preds, y, reduction="mean")
        accuracy = torch.sum(torch.argmax(preds, dim=1) == y) / len(y)
        loss_dict_no_ema.update({'val/loss': loss})
        loss_dict_no_ema.update({'val/accuracy': accuracy})

        loss_dict_ema = {}
        with self.ema_scope():
            # preds = self.model_ema.ema(x)
            preds = self.model(x)
            loss = nn.functional.cross_entropy(preds, y, reduction="mean")
            accuracy = torch.sum(torch.argmax(preds, dim=1) == y) / len(y)
            loss_dict_ema.update({'val/loss': loss})
            loss_dict_ema.update({'val/accuracy': accuracy})
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    # def p_losses(self, x_start, t, noise=None):
    #     loss, loss_dict = torch.zeros(1, device=self.device), {}
    #     if self.batch_classes is not None:
    #         prefix = 'train' if self.training else 'val'
    #         x = self.labeled_aug(x_start) if self.training else self.val_aug(x_start)
    #         self.batch_class_predictions = self.model(x)

    #         loss_classification = nn.functional.cross_entropy(
    #             self.batch_class_predictions, self.batch_classes)
    #         loss += loss_classification
    #         loss_dict.update(
    #             {f'{prefix}/loss_classification': loss_classification})
    #         loss_dict.update({f'{prefix}/loss': loss})
    #         accuracy = torch.sum(torch.argmax(
    #             self.batch_class_predictions, dim=1) == self.batch_classes) / len(self.batch_classes)
    #         loss_dict.update({f'{prefix}/accuracy': accuracy})
    #     if self.supervised_imgs is not None:

    #         prefix = 'train' if self.training else 'val'
    #         x = self.labeled_aug(self.supervised_imgs) if self.training else self.val_aug(self.supervised_imgs)
    #         preds = self.model(x)

    #         loss_classification = nn.functional.cross_entropy(
    #             preds, self.supervised_labels) * self.classification_loss_scale
    #         loss += loss_classification
    #         loss_dict.update(
    #             {f'{prefix}/loss_classification': loss_classification})
    #         loss_dict.update({f'{prefix}/loss': loss})
    #         accuracy = torch.sum(torch.argmax(preds, dim=1) == self.supervised_labels) / len(self.supervised_labels)
    #         loss_dict.update({f'{prefix}/accuracy': accuracy})

    #         self.supervised_imgs = None
    #         self.supervised_labels = None

    #     if not self.training:
    #         return loss, loss_dict
    #     prefix = 'train'

    #     with torch.no_grad():
    #         weakly_augmented = self.augmentation(self.raw_imgs).detach()
    #         weak_preds = self.model(
    #             weakly_augmented
    #         )

    #         weak_preds = nn.functional.softmax(weak_preds, dim=1).detach()
    #         pseudo_labels = weak_preds.argmax(dim=1)
    #         above_threshold_idx, = (
    #             weak_preds.max(dim=1).values > self.min_confidence
    #         ).nonzero(as_tuple=True)
    #         pseudo_labels = pseudo_labels[above_threshold_idx]

    #         loss_dict.update(
    #             {f'{prefix}/ssl_above_threshold': len(above_threshold_idx) / len(weak_preds)})
    #         loss_dict.update({f'{prefix}/ssl_max_confidence': weak_preds.max()})
    #         if len(above_threshold_idx) == 0:
    #             return loss, loss_dict

    #         strongly_augmented = self.strong_augmentation((self.raw_imgs[above_threshold_idx])).detach()
    #         strongly_augmented = self.cutout(strongly_augmented, level=1).detach()

    #     preds = self.model(
    #         strongly_augmented
    #     )
    #     ssl_loss = nn.functional.cross_entropy(preds, pseudo_labels)

    #     loss += ssl_loss * len(preds) / len(weak_preds)
    #     loss_dict.update({f'{prefix}/loss_ssl_classification': ssl_loss})
    #     loss_dict.update({f'{prefix}/loss': loss})
    #     accuracy = torch.sum(
    #         torch.argmax(preds, dim=1) == pseudo_labels) / len(pseudo_labels)
    #     loss_dict.update({f'{prefix}/ssl_accuracy': accuracy})

    #     return loss, loss_dict

    def on_train_batch_end(self, *args, **kwargs):
        self.scheduler.step()
        if self.use_ema:
            self.model_ema(self.model)
            # self.model_ema.update(self.model)
        return

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
