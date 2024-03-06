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
from torchvision.models import resnet50, densenet121
import importlib
from torchvision.models import DenseNet121_Weights

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
            weights=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ft_enc = False,
            ckpt_path = None,

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
        self.ft_enc = ft_enc
        self.ckpt_path = ckpt_path

        self.instantiate_modules()
        if self.ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=[])
        self.auroc_train = AUROC(num_classes=14)
        self.auroc_val = AUROC(num_classes=14)
        self.auroc_test = AUROC(num_classes=14)

        self.BCEweights = torch.Tensor(weights)

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
                nn.Linear(self.in_features, self.num_classes),
                )
        elif self.classifier_test_mode == "densenet":
            # ref impl https://github.com/zoogzog/chexnet/blob/master/DensenetModels.py
            self.densenet = densenet121(weights = DenseNet121_Weights.DEFAULT)
            self.num_classes = self.num_classes
            self.densenet.classifier = nn.Sequential(nn.Linear(self.in_features, self.num_classes))

        else:
            print('TEST NOT IMPLEMENTED')

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

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model
        if not self.ft_enc:
            self.first_stage_model.eval()
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
        elif self.classifier_test_mode == "densenet":
            out = self.densenet(imgs)
        return out
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
        scheduler = {
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min'),
        'monitor': "val/loss"
    }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss_dict = {}
        x, y = batch
        if len(x)<4:
            x = x.unsqueeze(1)
        if x.shape[-1]==3:
            x= x.permute(0,3,1,2)
        y_pred = self(x)

        loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y[:,:self.num_classes].float(), pos_weight=self.BCEweights.to(self.device)[:self.num_classes])
        accuracy = accuracy_score(y.cpu()[:,:14], y_pred.cpu()>=0.5)
        self.auroc_train.update(y_pred[:,:14], y[:,:14])
        self.log('train/auroc', self.auroc_train, on_step=False, on_epoch=True)
        loss_dict.update({'train/loss_classification': loss})
        loss_dict.update({'train/accuracy': accuracy})

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        return loss
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):
        self.log("global_step", self.global_step,
                    prog_bar=True, logger=True, on_step=False, on_epoch=False)
        x, y = batch
        if len(x)<4:
            x = x.unsqueeze(1)
        if x.shape[-1]==3:
            x= x.permute(0,3,1,2)
        y_pred = self(x)

        loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y.float()[:,:self.num_classes])

        if dataloader_idx == 0:
            accuracy = accuracy_score(y.cpu()[:,:self.num_classes], y_pred.cpu()>=0.5)
            self.auroc_val.update(y_pred[:,:14], y[:,:14])

            self.log('val/auroc', self.auroc_val, on_step=False, on_epoch=True, add_dataloader_idx=False)
            loss_dict = {"val/loss": loss, "val/accuracy": accuracy}
            self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, add_dataloader_idx=False)
            
        elif dataloader_idx == 1:
            accuracy = accuracy_score(y.cpu()[:,:self.num_classes], y_pred.cpu()>=0.5)
            self.auroc_test.update(y_pred[:,:14], y[:,:14])

            self.log('test/auroc', self.auroc_test, on_step=False, on_epoch=True, add_dataloader_idx=False)
            loss_dict = {"test/loss": loss, "test/accuracy": accuracy}
            self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, add_dataloader_idx=False)
            
        
        
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        
        x = batch[0]
        bs = x.shape[0]
        x = x.view(-1, 3,224, 224)
        y = batch[1]
        t = torch.zeros((x.shape[0],), device=self.device).long()
        
        y_pred = self(x)
        y_pred = nn.functional.sigmoid(y_pred)
        y_pred = y_pred.view(bs, 10, -1).mean(1)

        if y_pred.shape[1]!=y.shape[1]: #means one class less
            self.auroc_test.update(y_pred, y[:,:-1])
        else:
            self.auroc_test.update(y_pred[:,:-1], y[:,:-1])
        self.log('test/auroc', self.auroc_test, on_step=False, on_epoch=True, sync_dist=True)

