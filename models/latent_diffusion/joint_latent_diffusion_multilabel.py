import torch
from .joint_latent_diffusion_noisy_classifier import JointLatentDiffusionNoisyClassifier, JointLatentDiffusionNoisyAttention
from ..representation_transformer import RepresentationTransformer
import torch.nn as nn
from ..adjusted_unet import AdjustedUNet
import numpy as np
from sklearn.metrics import accuracy_score
from torchmetrics import AUROC
from ldm.modules.diffusionmodules.util import extract_into_tensor, noise_like
from tqdm import tqdm
from pytorch_msssim import ssim
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
#TODO remove unused imports!!!


class JointLatentDiffusionMultilabel(JointLatentDiffusionNoisyClassifier):
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 classifier_in_features,
                 classifier_hidden,
                 num_classes,
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
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            classifier_in_features=classifier_in_features,
            classifier_hidden=classifier_hidden,
            num_classes=num_classes,
            dropout=dropout,
            classification_loss_weight=classification_loss_weight,
            sample_grad_scale=100,
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
            *args,
            **kwargs
        )
        # self.x_start = None
        self.sampling_method = "conditional_to_x"
        self.num_classes = num_classes
        print('WARNING AUROC HARDCODEDDDD to 14 classes')
        self.auroc_train = AUROC(num_classes=14) #self.num_classes-1)
        self.auroc_val = AUROC(num_classes=14) #self.num_classes-1)
        self.BCEweights = torch.Tensor(weights)[:self.num_classes]

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        if self.classifier_lr:
            cl_lr = self.classifier_lr
            l = [{'params': [*self.model.parameters()], 'lr': lr},
                 {'params': [*self.classifier.parameters()], 'lr': cl_lr},]
        else:
            l = [{'params': [p for p in self.model.parameters()], 'lr': lr},]

        opt = torch.optim.AdamW(l, lr=0)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    def do_classification(self, x, t, y):
        unet: AdjustedUNet = self.model.diffusion_model
        representations = unet.just_representations(x, t, pooled=False)
        representations = self.transform_representations(representations)
        y_pred = self.classifier(representations)

        #skip last column if num classes < len(y_true) - we want to skip no findings label. BCE weights have to be adjusted!!!
        loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y.float()[:, :self.num_classes], pos_weight=self.BCEweights.to(self.device))
        accuracy = accuracy_score(y[:, :self.num_classes].cpu(), y_pred.cpu()>=0.5)
        return loss, accuracy, y_pred

    def train_classification_step(self, batch, loss):
        loss_dict = {}

        x, y = self.get_train_classification_input(batch, self.first_stage_key)
        t = torch.zeros((x.shape[0],), device=self.device).long()

        loss_classification, accuracy, y_pred = self.do_classification(x, t, y)

        if y_pred.shape[1]!=y.shape[1]: #means one class less in training
            self.auroc_train.update(y_pred, y[:,:-1])
        else:
            self.auroc_train.update(y_pred[:,:-1], y[:,:-1])

        loss += loss_classification * self.classification_loss_weight

        self.log('train/auroc', self.auroc_train, on_step=False, on_epoch=True)
        loss_dict.update({'train/loss_classification': loss_classification})
        loss_dict.update({'train/loss_full': loss})
        loss_dict.update({'train/accuracy': accuracy})
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = self.get_valid_classification_input(batch, self.first_stage_key)
        t = torch.zeros((x.shape[0],), device=self.device).long()
        x_diff, c = self.get_input(batch, self.first_stage_key)

        loss, loss_dict_no_ema = self(x_diff, c)
        loss_cls, accuracy, preds = self.do_classification(x, t, y)

        loss_dict_no_ema.update({'val/loss_classification': loss_cls})
        loss_dict_no_ema.update({'val/loss_full': loss + loss_cls})
        loss_dict_no_ema.update({'val/accuracy': accuracy})
        for i in range(preds.shape[1]):
            accuracy = torch.sum(
                ((torch.sigmoid(preds[:, i].detach()) >= 0.5).long() == y[:, i].long())
            ) / len(preds)
            loss_dict_no_ema.update({f'val/class_{i}_accuracy': accuracy})

        with self.ema_scope():
            loss, loss_dict_ema = self(x_diff, c)
            loss_cls, accuracy, y_pred = self.do_classification(x, t, y)

            if y_pred.shape[1]!=y.shape[1]: #means one class less
                self.auroc_val.update(y_pred, y[:,:-1])
            else:
                self.auroc_val.update(y_pred[:,:-1], y[:,:-1])

            self.log('val/auroc_ema', self.auroc_val, on_step=False, on_epoch=True)
            loss_dict_ema.update({'val/loss_classification': loss_cls})
            loss_dict_ema.update({'val/loss_full': loss + loss_cls})
            loss_dict_ema.update({'val/accuracy': accuracy})
            loss_dict_ema = {
                key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
            for i in range(y_pred.shape[1]):
                accuracy = torch.sum(
                    ((torch.sigmoid(y_pred[:, i].detach()) >= 0.5).long() == y[:, i].long())
                ) / len(y_pred)
                loss_dict_no_ema.update({f'val/class_{i}_accuracy': accuracy})

        self.log_dict(
            loss_dict_no_ema,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )
        self.log_dict(
            loss_dict_ema,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True
        )

    def p_mean_variance(self, x, c, t, clip_denoised: bool, original_img=None, pick_class = None, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None, return_pred_o=False):
        if self.sampling_method == "conditional_to_x":
            return self.grad_guided_p_mean_variance(x, t, clip_denoised, original_img =original_img, pick_class=pick_class, return_pred_o=return_pred_o)
        elif self.sampling_method == "conditional_to_repr":
            return self.grad_guided_p_mean_variance(x, t, clip_denoised)
        elif self.sampling_method == "unconditional":
            unet: AdjustedUNet = self.model.diffusion_model
            model_out = unet.just_reconstruction(x, t)
            if self.parameterization == "eps":
                x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
            elif self.parameterization == "x0":
                x_recon = model_out
            if clip_denoised:
                x_recon.clamp_(-1., 1.)

            model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
            if return_x0:
                return model_mean, posterior_variance, posterior_log_variance, x_recon
            else:
                return model_mean, posterior_variance, posterior_log_variance
        else:
            raise NotImplementedError("Sampling method not implemented")

    def grad_guided_p_mean_variance(self, x, t, clip_denoised: bool, original_img = None, pick_class=None, return_pred_o=False):
        if self.sampling_method == "conditional_to_x":
            model_out = self.guided_apply_model(x, t, original_img =original_img, pick_class=pick_class, return_pred_o=return_pred_o)
        elif self.sampling_method == "conditional_to_repr":
            model_out = self.guided_repr_apply_model(x, t)

        if return_pred_o:
            model_out, pred_o = model_out
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_pred_o:
            return model_mean, posterior_variance, posterior_log_variance, pred_o
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def guided_apply_model(self, x: torch.Tensor, t, original_img=None, pick_class='Cardiomegaly', return_pred_o=False):
        unet: AdjustedUNet = self.model.diffusion_model
        if not hasattr(self.model.diffusion_model, 'forward_input_blocks'):
            return unet.just_reconstruction(x, t)

        with torch.enable_grad():
            x = x.requires_grad_(True)
            pred_noise = unet.just_reconstruction(
                x, t, context=None)
            pred_x_start = (
                (x - extract_into_tensor(
                    self.sqrt_one_minus_alphas_cumprod, t, x.shape
                ) * pred_noise) /
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape)
            )
            representations = unet.just_representations(
                pred_x_start, t, context=None, pooled=False)
            pooled_representations = self.transform_representations(
                representations)
            pred = self.classifier(pooled_representations)

            representations_o = unet.just_representations(
                original_img, torch.zeros(len(original_img)).cuda(), context=None, pooled=False)
            pooled_representations_o = self.transform_representations(
                representations_o)
            pred_o = self.classifier(pooled_representations_o)

            x.retain_grad()
            #TODO here absolutely the worst, everything is hardcoded and chaged manually while running notebook!!!
            # TODO dont use it while training for logging
            cl_list = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion","Emphysema","Fibrosis", "Hernia","Infiltration", "Mass", "Nodule","Pleural_Thickening","Pneumonia","Pneumothorax","No Finding"]
            sample_classes = torch.zeros((x.shape[0], self.num_classes)).cuda()

            id_class = cl_list.index(pick_class)
            loss = +nn.functional.binary_cross_entropy_with_logits(pred[:,[id_class]], sample_classes[:,[id_class]], reduction="sum")

            loss.backward()
            s_t = self.sample_grad_scale * extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, (1,))[0]

            model_out = (pred_noise + s_t * x.grad).detach()
            #if t[0]<100:

                #print(cl_list)
                # print('all predictins for x pred start', nn.functional.sigmoid(pred))
                # print('all predictins for orig x', nn.functional.sigmoid(pred_o))
                #print(f'predicitons for x pred start, only class {cl_list[id_class]}', nn.functional.sigmoid(pred[:,[id_class]]))
                #print(f'predicitons for x original, only class {cl_list[id_class]}', nn.functional.sigmoid(pred_o[:,[id_class]]))
                # print('target label', sample_classes[:,[id_class]])


            #model_out = (pred_noise).detach()
        if return_pred_o:
            return model_out, pred_o
        else:
            return model_out

    @torch.no_grad()
    def guided_repr_apply_model(self, x, t):
        if not hasattr(self.model.diffusion_model, 'forward_input_blocks'):
            return self.apply_model(x, t)
        t_in = t

        emb = self.model.diffusion_model.get_timestep_embedding(x, t_in, None)
        unet: AdjustedUNet = self.model.diffusion_model

        with torch.enable_grad():
            representations = unet.forward_input_blocks(x, None, emb)
            for h in representations:
                h.retain_grad()

            pred_noise = unet.forward_output_blocks(x, None, emb, representations)
            pred_x_start = (
                (x - extract_into_tensor(
                    self.sqrt_one_minus_alphas_cumprod, t, x.shape
                ) * pred_noise) /
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape)
            )
            repr_x0 = unet.just_representations(
                pred_x_start, t, context=None, pooled=False)
            pooled_representations = self.transform_representations(
                repr_x0)
            class_predictions = self.classifier(pooled_representations)
            # loss = -torch.log(torch.gather(class_predictions, 1, self.sample_classes.unsqueeze(dim=1))).sum()
            loss = -nn.functional.cross_entropy(class_predictions, self.sample_classes, reduction="sum")
            loss.backward()
            s_t = self.sample_grad_scale * extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, (1,))[0]
            representations = [(h + self.sample_grad_scale * h.grad * s_t).detach() for h in representations]

        model_out = unet.forward_output_blocks(x, None, emb, representations)
        return model_out

    @torch.no_grad()
    def p_sample(self, x, c, t, original_img=None, pick_class=None, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False, return_pred_o=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, original_img=original_img, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0, pick_class=pick_class,return_pred_o=return_pred_o,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)



        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        elif return_pred_o:
            model_mean, _, model_log_variance, pred_o = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        elif return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        elif return_pred_o:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, pred_o
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, original_img=None, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, pick_class=None, start_T=None,
                      log_every_t=None, return_pred_o=False):

        if return_pred_o:
            print("[WARNING] models/latent_diffusion/joint_latent_diffusion_multilabel.py, p_sample_loop: return_pred_o not implemented for every case" )
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,original_img=original_img,
                                quantize_denoised=quantize_denoised, pick_class=pick_class, return_pred_o=return_pred_o)

            if return_pred_o:
                img, pred_o = img
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_pred_o:
            print(['[WARNING] Return_pred_o cannot return intermediates - not impemented!'])
            return img,pred_o
        if return_intermediates:
            return img, intermediates
        return img


class JointLatentDiffusionMultilabelAttention(JointLatentDiffusionMultilabel):
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 attention_config,
                 classifier_in_features,
                 classifier_hidden,
                 num_classes,
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
        self.classifier = RepresentationTransformer(**attention_config)

    def transform_representations(self, representations):
        return representations
