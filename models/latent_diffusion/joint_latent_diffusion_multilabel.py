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
        self.gradient_guided_sampling = False
        self.sampling_method = "conditional_to_x"
        self.num_classes = num_classes
        print('WARNING AUROC HARDCODEDDDD to 14 classes')
        self.auroc_train = AUROC(num_classes=14) #self.num_classes-1)
        self.auroc_val = AUROC(num_classes=14) #self.num_classes-1)
        self.BCEweights = torch.Tensor(weights)[:self.num_classes]
     
    
    def do_classification(self, x, t, y):
        unet: AdjustedUNet = self.model.diffusion_model
        representations = unet.just_representations(x, t, pooled=False)
        representations = self.transform_representations(representations)
        y_pred = self.classifier(representations)
        
        #skip last column if num classes < len(y_true) - we want to skip no findings label. BCE weights have to be adjusted!!!
        loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y.float()[:,:self.num_classes], pos_weight=self.BCEweights.to(self.device))
        accuracy = accuracy_score(y[:,:self.num_classes].cpu(), y_pred.cpu()>=0.5)
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
        loss_cls, accuracy, _ = self.do_classification(x, t, y)            

        loss_dict_no_ema.update({'val/loss_classification': loss_cls})
        loss_dict_no_ema.update({'val/loss_full': loss + loss_cls})
        loss_dict_no_ema.update({'val/accuracy': accuracy})


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


    def p_mean_variance(self, x, c, t, clip_denoised: bool, original_img=None, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        if self.sampling_method == "conditional_to_x":
            return self.grad_guided_p_mean_variance(x, t, clip_denoised, original_img =original_img)
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
            return model_mean, posterior_variance, posterior_log_variance
        else:
            raise NotImplementedError("Sampling method not implemented")

    def grad_guided_p_mean_variance(self, x, t, clip_denoised: bool, original_img = None):
        if self.sampling_method == "conditional_to_x":
            model_out = self.guided_apply_model(x, t, original_img =original_img)
        elif self.sampling_method == "conditional_to_repr":
            model_out = self.guided_repr_apply_model(x, t)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def guided_apply_model(self, x: torch.Tensor, t, original_img=None):
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

            x.retain_grad()
            sample_classes = torch.zeros((x.shape[0], self.num_classes)).cuda()
            sample_classes[:, -1] = 1
            loss = nn.functional.binary_cross_entropy_with_logits(pred, sample_classes, reduction="sum")
            cl_list = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion","Emphysema","Fibrosis", "Hernia","Infiltration", "Mass", "Nodule","Pleural_Thickening","Pneumonia","Pneumothorax","No Finding"]
            print([*zip(nn.functional.sigmoid(pred[0]), cl_list)])
            loss.backward()
            s_t = self.sample_grad_scale * extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, (1,))[0]
            #model_out = (pred_noise + s_t * x.grad).detach()
            model_out = (pred_noise).detach()

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
    def p_sample(self, x, c, t, original_img=None, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, original_img=original_img, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, original_img=None, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

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
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img


