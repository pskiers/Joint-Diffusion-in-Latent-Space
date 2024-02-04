import torch
import torch.nn as nn
import pytorch_lightning as pl
from ..latent_diffusion.joint_latent_diffusion_multilabel import JointLatentDiffusionMultilabel
from ldm.modules.diffusionmodules.util import timestep_embedding
import importlib
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

class MultilabelClassifierOnLatentDiffusion(JointLatentDiffusionMultilabel):
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
            *args,
            **kwargs
            )

    def configure_optimizers(self):
            lr = self.learning_rate
            weight_decay = self.weight_decay
            params = list(self.model.parameters())
            if self.cond_stage_trainable:
                print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
                params = params + list(self.cond_stage_model.parameters())
            if self.learn_logvar:
                print('Diffusion model optimizing logvar')
                params.append(self.logvar)
            #TODO parametrize
            #opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
            opt = torch.optim.AdamW([
                #{'params':[*self.first_stage_model.parameters()], 'lr' : lr/50},
                {'params':[*self.model.parameters()], 'lr':lr, 'weight_decay':weight_decay}
                ])
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
            
            # if True:
            #     print("Setting up OneCycleLR scheduler...")
            #     scheduler = [
            #         {
            #             'scheduler': OneCycleLR(opt, max_lr=1e-3, total_steps=self.trainer.max_steps),

            #         }]
            #     return [opt], scheduler

            return opt

    # override to allow encoder finetuning. TODO parametrize
    # def instantiate_first_stage(self, config):
    #     self.first_stage_model = instantiate_from_config(config)
    
    def training_step(self, batch, batch_idx):
        loss_dict = {}

        x, y = self.get_train_classification_input(batch, self.first_stage_key)
        t = torch.zeros((x.shape[0],), device=self.device).long()

        loss_classification, accuracy, y_pred = self.do_classification(x, t, y)

        self.auroc_train.update(y_pred[:,:-1], y[:,:-1])
        self.log('train/auroc', self.auroc_train, on_step=False, on_epoch=True)

        loss_dict.update({'train/loss_classification': loss_classification})
        loss_dict.update({'train/accuracy': accuracy})

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        return loss_classification


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss_dict_no_ema ={}
        x, y = self.get_valid_classification_input(batch, self.first_stage_key)
        t = torch.zeros((x.shape[0],), device=self.device).long()

        loss_cls, accuracy, _ = self.do_classification(x, t, y)
        loss_dict_no_ema.update({'val/loss_classification': loss_cls})
        loss_dict_no_ema.update({'val/accuracy': accuracy})


        with self.ema_scope():
            loss_dict_ema = {}
            loss_cls, accuracy, y_pred = self.do_classification(x, t, y)
            self.auroc_val.update(y_pred[:,:-1], y[:,:-1])
            self.log('val/auroc_ema', self.auroc_val, on_step=False, on_epoch=True)

            loss_dict_ema.update({'val/loss_classification': loss_cls})
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


