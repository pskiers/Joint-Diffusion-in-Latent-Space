import math
from contextlib import contextmanager
import torch
import torch.nn as nn
import torchvision as tv
import pytorch_lightning as pl
from ldm.util import count_params
from ..utils import FixMatchEma, interleave, de_interleave
from ..wide_resnet import Wide_ResNet
from torchmetrics import AUROC


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
        # self.model = Wide_ResNet(depth=28, num_classes=10, widen_factor=8, drop_rate=0)
        # self.model = tv.models.densenet121(weights="DEFAULT")
        self.model = tv.models.densenet121()
        self.model.classifier = nn.Linear(in_features=1024, out_features=8)
        count_params(self.model, verbose=True)
        self.model_ema = FixMatchEma(self.model, decay=0.999)
        self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
        self.img_key = img_key
        self.label_key = label_key
        self.unsup_img_key = unsup_img_key
        self.use_ema = True
        self.scheduler = None
        self.auroc_val = AUROC(num_classes=8) #self.num_classes-1)

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
        y = batch[0][self.label_key]
        _, weak_img, strong_img = batch[1][0]
        return x, y, weak_img, strong_img

    def get_val_input(self, batch):
        x = batch[self.img_key]
        y = batch[self.label_key]
        return x, y

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
        # loss_classification = nn.functional.binary_cross_entropy_with_logits(preds_x, y.float(), reduction="mean")
        loss += loss_classification
        accuracy = torch.sum(torch.argmax(preds_x, dim=1) == y) / len(y)
        # accuracy = torch.sum((preds_x.detach() >= 0).long() == y) / torch.numel(y)
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
            (torch.argmax(preds_strong, dim=1) == targets_u) * mask
        ) / mask.sum() if mask.sum() > 0 else 0
        loss_dict.update(
                {'train/ssl_above_threshold': mask.mean().item()})
        loss_dict.update({'train/ssl_max_confidence': mask.max().item()})
        loss_dict.update({'train/loss_ssl_classification': ssl_loss})
        loss_dict.update({'train/loss': loss})
        loss_dict.update({'train/ssl_accuracy': accuracy})

        # pseudo_label = torch.sigmoid(preds_weak.detach())
        # mask = torch.logical_or(
        #     pseudo_label >= self.min_confidence,
        #     pseudo_label <= (-self.min_confidence + 1)
        # ).float()
        # targets_u = (pseudo_label >= 0.5).float()
        # ssl_loss = (nn.functional.binary_cross_entropy_with_logits(
        #     preds_strong, targets_u, reduction='none') * mask).mean()
        # loss += ssl_loss
        # accuracy = torch.sum(
        #     ((torch.sigmoid(preds_strong) >= 0.5).long() == targets_u.long()) * mask
        # ) / mask.sum() if mask.sum() > 0 else 0
        # loss_dict.update({'train/ssl_above_threshold': mask.mean().item()})
        # loss_dict.update({'train/loss_ssl_classification': ssl_loss})
        # loss_dict.update({'train/loss': loss})
        # loss_dict.update({'train/ssl_accuracy': accuracy})

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
        # loss = nn.functional.binary_cross_entropy_with_logits(preds, y.float(), reduction="mean")
        accuracy = torch.sum(torch.argmax(preds, dim=1) == y) / len(y)
        # accuracy = torch.sum((preds.detach() >= 0).long() == y) / torch.numel(y)
        loss_dict_no_ema.update({'val/loss': loss})
        loss_dict_no_ema.update({'val/accuracy': accuracy})

        loss_dict_ema = {}
        with self.ema_scope():
            preds = self.model(x)
            loss = nn.functional.cross_entropy(preds, y, reduction="mean")
            # loss = nn.functional.binary_cross_entropy_with_logits(preds, y.float(), reduction="mean")
            accuracy = torch.sum(torch.argmax(preds, dim=1) == y) / len(y)
            # accuracy = torch.sum((preds.detach() >= 0).long() == y) / torch.numel(y)
            loss_dict_ema.update({'val/loss': loss})
            loss_dict_ema.update({'val/accuracy': accuracy})
            # self.auroc_val.update(preds, y.int())
            # self.log('val/auroc_ema', self.auroc_val, on_step=False, on_epoch=True)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        self.scheduler.step()
        if self.use_ema:
            self.model_ema(self.model)
        return


class MeanMatch(FixMatch):
    def __init__(self,
                 min_confidence=0.95,
                 mu=7,
                 batch_size=64,
                 img_key=0,
                 label_key=1,
                 unsup_img_key=0,
                 monitor="val/loss_ema",
                 ckpt_path=None,
                 *args,
                 **kwargs):
        super().__init__(
            min_confidence,
            mu,
            batch_size,
            img_key,
            label_key,
            unsup_img_key,
            monitor,
            ckpt_path,
            *args, **kwargs
        )
        self.num_classes = 10

    def get_train_input(self, batch):
        x = batch[0][self.img_key]
        y = batch[0][self.label_key]
        img, weak_img, strong_img = batch[1][0]
        _, _, c, h, w = img.shape
        del img
        return (
            x, y,
            weak_img.view(-1, c, h, w),
            strong_img.view(-1, c, h, w)
        )

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

        b, c = preds_weak.shape
        reshape_weak = preds_weak.view(
            b // self.num_classes, self.num_classes, c)
        weak_clone = reshape_weak.clone()
        weak_clone[:] = torch.sum(
            reshape_weak, dim=1, keepdim=True) / self.num_classes
        weak_clone = weak_clone.view(b, c)

        pseudo_label = torch.softmax(weak_clone.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.min_confidence).float()
        ssl_loss = (nn.functional.cross_entropy(
            preds_strong, targets_u, reduction='none') * mask).mean()
        loss += ssl_loss
        accuracy = torch.sum(
            (torch.argmax(preds_strong, dim=1) == targets_u) * mask
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
