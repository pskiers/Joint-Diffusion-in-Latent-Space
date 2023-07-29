import math
import torch
import torch.nn as nn
from einops import rearrange
import kornia as K
from ldm.models.diffusion.ddpm import DDPM
from ldm.util import default
from .ssl_joint_diffusion import SSLJointDiffusion
from ..representation_transformer import RepresentationTransformer
from ..baselines.fixmatch import FixMatchEma, interleave, de_interleave


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
            if isinstance(weak_rep, list): # TODO refactor this shit
                weak_rep = self.transform_representations(weak_rep)
                weak_preds = nn.functional.softmax(
                    self.classifier(weak_rep), dim=1).detach()
            else:
                weak_preds = nn.functional.softmax(weak_rep, dim=1).detach()
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
        if isinstance(strong_rep, list): # TODO refactor this shit
            strong_rep = self.transform_representations(strong_rep)
            preds = self.classifier(strong_rep)
        else:
            preds = strong_rep
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


class DiffMatchAttention(DiffMatch):
    def __init__(self, attention_config, *args, **kwargs):
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


class DiffMatchFixed(DDPM):
    def __init__(self,
                 min_confidence=0.95,
                 mu=7,
                 batch_size=64,
                 img_key=0,
                 label_key=1,
                 unsup_img_key=0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.min_confidence = min_confidence
        self.mu = mu
        self.batch_size = batch_size
        self.model_ema = FixMatchEma(self.model, decay=0.999)
        self.img_key = img_key
        self.label_key = label_key
        self.unsup_img_key = unsup_img_key
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
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out, _ = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def get_train_input(self, batch):
        x = batch[0][self.img_key]
        y = batch[0][self.label_key]
        weak_img, strong_img = batch[1][0]
        return x, y, weak_img, strong_img

    def get_val_input(self, batch):
        x = batch[self.img_key]
        y = batch[self.label_key]
        return x, y

    def training_step(self, batch, batch_idx):
        x, y, weak_img, strong_img = self.get_train_input(batch)
        loss, loss_dict = self(weak_img)

        inputs = interleave(
            torch.cat((x, weak_img, strong_img)), 2*self.mu+1)
        logits = self.model.diffusion_model.just_representations(inputs)
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
            (torch.argmax(preds_strong, dim=1) == targets_u) * mask
        ) / mask.sum() if mask.sum() > 0 else 0
        loss_dict.update(
                {'train/ssl_above_threshold': mask.mean().item()})
        loss_dict.update({'train/ssl_max_confidence': mask.max().item()})
        loss_dict.update({'train/loss_ssl_classification': ssl_loss})
        loss_dict.update({'train/loss': loss})
        loss_dict.update({'train/ssl_accuracy': accuracy})

        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = self.get_val_input(batch)
        _, loss_dict_no_ema = self(x)
        preds = self.model.diffusion_model.just_representations(x)
        loss = nn.functional.cross_entropy(preds, y, reduction="mean")
        accuracy = torch.sum(torch.argmax(preds, dim=1) == y) / len(y)
        loss_dict_no_ema.update({'val/loss': loss})
        loss_dict_no_ema.update({'val/accuracy': accuracy})
        with self.ema_scope():
            x, y = self.get_val_input(batch)
            _, loss_dict_ema = self(x)
            preds = self.model.diffusion_model.just_representations(x)
            loss = nn.functional.cross_entropy(preds, y, reduction="mean")
            accuracy = torch.sum(torch.argmax(preds, dim=1) == y) / len(y)
            loss_dict_ema.update({'val/loss': loss})
            loss_dict_ema.update({'val/accuracy': accuracy})
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        self.scheduler.step()
        if self.use_ema:
            self.model_ema(self.model)
            # self.model_ema.update(self.model)
        return

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out, _ = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
