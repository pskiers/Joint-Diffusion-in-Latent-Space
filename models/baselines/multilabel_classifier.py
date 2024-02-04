import torch
from ..representation_transformer import RepresentationTransformer
import torch.nn as nn
from ..adjusted_unet import AdjustedUNet
import numpy as np
from sklearn.metrics import accuracy_score
from torchmetrics import AUROC
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution


import torch
import torch.nn as nn
import pytorch_lightning as pl
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.diffusionmodules.util import timestep_embedding
from torchvision.models import resnet50
import importlib

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

class MultilabelClassifier(pl.LightningModule):
    def __init__(
            self,
            image_size: int,
            channels: int,
            first_stage_config,
            num_classes: int,
            in_features: int,
            monitor,
            dropout: float =0,
            learning_rate: float=0.00001,
            weight_decay: float = 0,
            classifier_test_mode: str = "encoder_resnet",
        ) -> None:
        super().__init__()

        self.classifier_test_mode = classifier_test_mode
        self.channels = channels
        self.num_classes = num_classes
        self.in_features = in_features
        self.dropout = dropout
        self.first_stage_config = first_stage_config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.instantiate_modules()
        self.auroc_train = AUROC(num_classes=num_classes-1)
        self.auroc_val = AUROC(num_classes=num_classes-1)
        if monitor is not None:
            self.monitor = monitor

    def instantiate_modules(self):
        if self.classifier_test_mode == "encoder_resnet":
                self.instantiate_first_stage(self.first_stage_config)
                self.resnet = resnet50()
                self.resnet.conv1 = nn.Conv2d(self.channels, 64, kernel_size=3, stride=1, padding=2, bias=False)
                self.resnet.maxpool = nn.Identity()
                self.num_classes = self.num_classes
                self.resnet.fc = nn.Sequential(
                    nn.Dropout(p=self.dropout),
                    nn.Linear(self.in_features, self.num_classes)
                    )
        elif self.classifier_test_mode == "encoder_linear":
            self.instantiate_first_stage(self.first_stage_config)
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.in_features, self.in_features//8),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.in_features//8, self.num_classes)
                )
        elif self.classifier_test_mode == "resnet":
            self.resnet = resnet50()
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            #for smaller imgs #self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            self.resnet.maxpool = nn.Identity()
            self.num_classes = self.num_classes
            self.resnet.fc = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.in_features, self.in_features//8),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.in_features//8, self.num_classes),
                )
        else:
            print('TEST NOT IMPLEMENTED')


    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode_first_stage(self, x):
        encoder_posterior = self.first_stage_model.encode(x)
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return  z 

    def forward(self, imgs: torch.Tensor):
        if self.classifier_test_mode == "encoder_resnet":
            out = self.encode_first_stage(imgs)
            out = self.resnet(out)
        elif self.classifier_test_mode == "encoder_linear":
            out = self.encode_first_stage(imgs)
            out = self.fc(out)
        elif self.classifier_test_mode == "resnet":
            out = self.resnet(imgs)
        return out
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)

    def training_step(self, batch, batch_idx):
        loss_dict = {}
        x, y = batch
        x = x.unsqueeze(1)
        y_pred = self(x)

        loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y.float())
        accuracy = accuracy_score(y.cpu(), y_pred.cpu()>=0.5)
        self.auroc_train.update(y_pred[:,:-1], y[:,:-1])
        self.log('train/auroc', self.auroc_train, on_step=False, on_epoch=True)
        loss_dict.update({'train/loss_classification': loss})
        loss_dict.update({'train/accuracy': accuracy})

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.unsqueeze(1)
        y_pred = self(x)

        loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y.float())
        accuracy = accuracy_score(y.cpu(), y_pred.cpu()>=0.5)
        self.auroc_val.update(y_pred[:,:-1], y[:,:-1])

        self.log('val/auroc', self.auroc_val, on_step=False, on_epoch=True)
        loss_dict = {"val/loss": loss, "val/accuracy": accuracy}
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=False, on_epoch=False)
        
        return loss_dict
