import torch
from ..representation_transformer import RepresentationTransformer
import torch.nn as nn
from ..adjusted_unet import AdjustedUNet
import numpy as np
from sklearn.metrics import accuracy_score
from torchmetrics import AUROC
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from models.pl_utils import PLUL
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.diffusionmodules.util import timestep_embedding
from torchvision.models import resnet50, densenet121
import importlib
from torchvision.models import DenseNet121_Weights
from datasets.chest_xray_acpl import ChestACPLDataloader, ChestACPLDataset
from easydict import EasyDict as edict
from tqdm import tqdm
import time
from types import SimpleNamespace

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

def lr_lambda(epoch: int):
        if epoch > 15:
            return 0.1
        else:
            return 1.0
class MultilabelClassifierACPL(pl.LightningModule):
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
            classifier_test_mode: str = "densenet",
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
        self.acpl_loop = 0

        self.instantiate_modules()
        if self.ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=[])
        #self.auroc_train = AUROC(num_classes=14)
        self.auroc_val = AUROC(num_classes=14)
        self.auroc_test = AUROC(num_classes=14)

        self.BCEweights = torch.Tensor(weights)

        if monitor is not None:
            self.monitor = monitor
        
    def instantiate_modules(self):
        if self.classifier_test_mode == "encoder_resnet":
                raise "Not a baseline, not tested, pls dont use it"
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
            raise "Not a baseline, not tested, pls dont use it"
            self.instantiate_first_stage(self.first_stage_config)
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.in_features, self.in_features//8),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.in_features//8, self.num_classes)
                )
        elif self.classifier_test_mode == "resnet":
            raise "Not a baseline, not tested, pls dont use it"
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
            #out = self.densenet(imgs)
            repr = self.densenet.features(imgs)
            repr = F.relu(repr, inplace=True)
            repr = F.adaptive_avg_pool2d(repr, (1, 1))
            repr = torch.flatten(repr, 1)
            out = self.densenet.classifier(repr)
        return out
    
    def forward_with_repr(self, imgs: torch.Tensor):
        if self.classifier_test_mode == "encoder_resnet":
            raise NotImplementedError
        elif self.classifier_test_mode == "encoder_linear":
            raise NotImplementedError
        elif self.classifier_test_mode == "resnet":
            raise NotImplementedError
        elif self.classifier_test_mode == "densenet":
            repr = self.densenet.features(imgs)
            repr = F.relu(repr, inplace=True)
            repr = F.adaptive_avg_pool2d(repr, (1, 1))
            repr = torch.flatten(repr, 1)
            out = self.densenet.classifier(repr)

        return out, F.normalize(repr, dim=-1, p=2)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay, eps=0.1, betas=(0.9, 0.99))
        scheduler = {
        'scheduler': LambdaLR(optimizer, lr_lambda=lr_lambda),
    }
        return [optimizer], [scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):
        loss_dict = {}
        x, y, _, _ = batch
        if len(x)<4:
            x = x.unsqueeze(1)
        if x.shape[-1]==3:
            x= x.permute(0,3,1,2)
        y_pred = self(x)

        loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y[:,:self.num_classes].float(), pos_weight=self.BCEweights.to(self.device)[:self.num_classes])
        accuracy = 0 #accuracy_score(y.cpu()[:,:14], y_pred[:,:14].cpu()>=0.5)
        #self.auroc_train.update(y_pred[:,:14], y[:,:14])
        #self.log('train/auroc', self.auroc_train, on_step=False, on_epoch=True)
        loss_dict.update({'train/loss_classification': loss})
        loss_dict.update({'train/accuracy': accuracy})

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx,):
        self.log("global_step", self.global_step,
                    prog_bar=True, logger=True, on_step=True, on_epoch=False)
        x, y, _, _ = batch
        if len(x)<4:
            x = x.unsqueeze(1)
        if x.shape[-1]==3:
            x= x.permute(0,3,1,2)
        y_pred = self(x)

        loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y.float()[:,:self.num_classes])

        # if dataloader_idx == 0:
        #     accuracy = accuracy_score(y.cpu()[:,:self.num_classes], y_pred.cpu()>=0.5)
        #     self.auroc_val.update(y_pred[:,:14], y[:,:14])

        #     self.log('val/auroc', self.auroc_val, on_step=False, on_epoch=True, add_dataloader_idx=False)
        #     loss_dict = {"val/loss": loss, "val/accuracy": accuracy}
        #     self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, add_dataloader_idx=False)
            
        # elif dataloader_idx == 1:
        accuracy = accuracy_score(y.cpu()[:,:self.num_classes], y_pred.cpu()>=0.5)
        self.auroc_test.update(y_pred[:,:14], y[:,:14])

        self.log('test/auroc', self.auroc_test, on_step=False, on_epoch=True, add_dataloader_idx=False)
        loss_dict = {"test/loss": loss, "test/accuracy": accuracy}
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, add_dataloader_idx=False)
        
        
    
    # @torch.no_grad()
    # def test_step(self, batch, batch_idx):
        
    #     x = batch[0]
    #     bs = x.shape[0]
    #     x = x.view(-1, 3,224, 224)
    #     y = batch[1]
    #     t = torch.zeros((x.shape[0],), device=self.device).long()
        
    #     y_pred = self(x)
    #     y_pred = nn.functional.sigmoid(y_pred)
    #     y_pred = y_pred.view(bs, 10, -1).mean(1)

    #     if y_pred.shape[1]!=y.shape[1]: #means one class less
    #         self.auroc_test.update(y_pred, y[:,:-1])
    #     else:
    #         self.auroc_test.update(y_pred[:,:-1], y[:,:-1])
    #     self.log('test/auroc', self.auroc_test, on_step=False, on_epoch=True, sync_dist=True)

    
    def on_train_epoch_end(self) -> None:
        # we have psudolabel-enhanced trainings in between epoch ends. That;s why we have strange if-conditions
        if (self.current_epoch+1)%10==0 and (self.current_epoch+1)>10:    
            print('current epoch, curr dataloader len', self.current_epoch, len(self.trainer.datamodule.train_dataloader()))
            
            if dist.get_rank()==0:
                self.acpl_actions_before_training_loop()
                objects = [self.trainer.labeled_loader]
            else:
                objects = [None]
            dist.barrier()
            dist.broadcast_object_list(objects, src=0)
            new_labeled_loader = objects[0]
            print("#%&$&%&%&%&%&%&%%&%&%&%&%", self.trainer.global_rank, len(new_labeled_loader))
            self.trainer.datamodule.update_train_loader(new_labeled_loader)
        elif (self.current_epoch)%10==0 and self.acpl_loop>0: 
            if dist.get_rank()==0:
                self.acpl_actions_after_training_loop()
        return 
    
    def mock_acpl_args(self):
        args = {'ds_mixup': True, 
                'num_gmm_sets': 3,
                 "sel":2, 
                 "topk": 200, 
                 "gpu": self.device.index,
                 "label_ratio":2, 
                 "runtime": 1}
        self.mock_args = SimpleNamespace(**args)

    def acpl_actions_before_training_loop(self):
        if self.acpl_loop==0:
            # we want it only once, in original ACPL it was before acpl loops
            self.trainer.anchor = self._anchor_ext()
            if dist.get_rank() == 0:
                torch.save(self.trainer.anchor, f"{self.trainer.logdir}/anchor0.pth.tar")

        print(f"Finished {self.acpl_loop} acpl loop (Loop 0 had no pseudolabels, PseudoLabel started in loop 1). Now KNN and pseudolabeling for the next loop.")
        self.acpl_loop+=1
        
        unlabel = self._anchor_sim()
        if dist.get_rank() == 0:
            torch.save(unlabel, f"{self.trainer.logdir}/ugraph{self.acpl_loop}.pth.tar")

        # Build local-KNN graph
        #dist.barrier()
        if dist.get_rank() == 1:
            time.sleep(10)
        print(f"Rank {dist.get_rank()} start graph building")
        self.mock_acpl_args() #we need some args typical for acpl
        lpul = PLUL(
            self.trainer.anchor,
            unlabel,
            self.mock_args,
            self.trainer.logdir,
            ds_mixup=self.mock_args.ds_mixup,
            loop=self.acpl_loop,
            num_gmm_sets=self.mock_args.num_gmm_sets,
        )
        print(f"Rank {dist.get_rank()} plul initialized")
        # label
        l_sel_idxs, l_sel_p = lpul.get_new_label()

        # unlabel
        u_sel_idxs = lpul.get_new_unlabel()

        # anchor
        anchor_idxs = lpul.anchor_purify()
        a_sel_idxs, a_sel_p = lpul.get_new_anchor(anchor_idxs)

        self.trainer.labeled_dataset.x_add_pl(l_sel_p, l_sel_idxs)
        self.trainer.labeled_dataset.u_update_pl(l_sel_idxs)

        self.trainer.anchor_dataset.x_add_pl(a_sel_p, a_sel_idxs)
        # self.anchor_dataset.u_update_pl(a_sel_idxs)

        self.trainer.unlabeled_dataset.u_update_pl(u_sel_idxs)
        print(
            f"Rank {dist.get_rank()} We prepared for loop {self.acpl_loop} following datasets: \
            label size {len(self.trainer.labeled_dataset)}, unlabel size {len(self.trainer.unlabeled_dataset)}"
        )

        print('Increment loop value, new val', self.acpl_loop)
        del lpul, self.trainer.anchor, unlabel
        #dist.barrier()
        (self.trainer.labeled_loader, self.trainer.labeled_dataset, self.trainer.labeled_sampler,) = self.trainer.acpl_loader.run(
            "labeled",
            dataset=self.trainer.labeled_dataset,
            ratio=self.mock_args.label_ratio,
            runtime=self.mock_args.runtime,
        )

        (self.trainer.unlabeled_loader, self.trainer.unlabeled_dataset, self.trainer.unlabeled_sampler,) = self.trainer.acpl_loader.run(
            "unlabeled",
            dataset=self.trainer.unlabeled_dataset,
            ratio=self.mock_args.label_ratio,
            runtime=self.mock_args.runtime,
        )

        self.trainer.anchor_loader, self.trainer.anchor_dataset, self.trainer.sampler = self.trainer.acpl_loader.run(
            "anchor",
            dataset=self.trainer.anchor_dataset,
            ratio=self.mock_args.label_ratio,
            runtime=self.mock_args.runtime,
        )
        
        
    
    def acpl_actions_after_training_loop(self):
        #dist.barrier()
        #self._ck_load(args, self.best_iter, self.best_loop)

        if dist.get_rank() == 0:
            print(
                f"Current pseudo Anchor size {len(self.trainer.anchor_dataset)}, Last loop anchor size {self.trainer.last_loop_anchor_len}"
            )
        # verify pseudo anchors
        pa_pack = self._anchor_ext()  # last loop anchors + info
        self.trainer.last_loop_anchor_len = len(self.trainer.anchor_dataset)
        self.trainer.anchor = pa_pack
        if dist.get_rank() == 0:
            torch.save(self.trainer.anchor, f"{self.trainer.logdir}/anchor{self.acpl_loop}.pth.tar")


    def _anchor_sim(self):
        print('###########ANCHOR SIM START TEST RANK', dist.get_rank())
        self.eval()

        u_gts, u_idxs = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        u_preds1 = torch.tensor([]).to(self.device)
        u_embed1 = torch.tensor([]).to(self.device)
        u_logits1 = torch.tensor([]).to(self.device)
        u_paths = torch.tensor([]).to(self.device)

        
        with torch.no_grad():
            
            print('$$$$$$$$$$$$ WE IN LOOP ANCHOR SIM START TEST RANK', dist.get_rank())
            # we operate on unlabeled dataloader
            for batch_idx, batch in enumerate(tqdm(self.trainer.unlabeled_loader)):
                (inputs, labels, item, input_path) = self.transfer_batch_to_device(batch, self.device, dataloader_idx=1)
                outputs1, feat1 = self.forward_with_repr(inputs)
                
                # inputs, labels = inputs.cuda(), labels.cuda(args.gpu)
                # item = item.cuda(args.gpu)
                # input_path = input_path.cuda(args.gpu)

                outputs1, feat1 = self.forward_with_repr(inputs)
                u_embed1 = torch.cat((u_embed1, feat1))

                u_idxs = torch.cat((u_idxs, item))
                u_paths = torch.cat((u_paths, input_path))
                u_gts = torch.cat((u_gts, labels))

                u_preds1 = torch.cat((u_preds1, torch.sigmoid(outputs1)))
                u_logits1 = torch.cat((u_logits1, outputs1))

        length = len(self.trainer.unlabeled_loader.dataset)
        u_idxs = self.distributed_concat(u_idxs.contiguous(), length)
        u_gts = self.distributed_concat(u_gts.contiguous(), length)
        u_preds1 = self.distributed_concat(u_preds1.contiguous(), length)
        u_logits1 = self.distributed_concat(u_logits1.contiguous(), length)
        u_embed1 = self.distributed_concat(u_embed1.contiguous(), length)
        u_paths = self.distributed_concat(u_paths.contiguous(), length)
        print('###########ANCHOR SIM END TEST RANK', dist.get_rank(), length)

        return edict(
            {
                "idxs": u_idxs,
                "gts": u_gts,
                "p1": u_preds1,
                "embed1": u_embed1,
                "logits1": u_logits1,
                "path": u_paths,
            }
        )

    def _anchor_ext(self):
        print('###########ANCHOR EXT START TEST RANK', dist.get_rank())

        self.eval()
        p1, logits1, embed1, gts, idxs = (
            torch.tensor([]).to(self.device),
            torch.tensor([]).to(self.device),
            torch.tensor([]).to(self.device),
            torch.tensor([]).to(self.device),
            torch.tensor([]).to(self.device),
        )
        with torch.no_grad():

            print('$$$$$$$$$$$$ WE IN LOOP ANCHOR EXT START TEST RANK', dist.get_rank())

            for batch_idx, batch in enumerate(self.trainer.anchor_loader):
                (inputs, labels, item, _)  = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
                outputs1, feat1 = self.forward_with_repr(inputs)
                #item = torch.from_numpy(item.numpy())#.cuda(args.gpu)
                embed1 = torch.cat((embed1, feat1))
                idxs = torch.cat((idxs, item))
                gts = torch.cat((gts, labels))
                logits1 = torch.cat((logits1, outputs1))
                p1 = torch.cat((p1, torch.sigmoid(outputs1)))
        length = len(self.trainer.anchor_loader.dataset)
        p1 = self.distributed_concat(p1.contiguous(), length)
        logits1 = self.distributed_concat(logits1.contiguous(), length)
        embed1 = self.distributed_concat(embed1.contiguous(), length)
        gts = self.distributed_concat(gts.contiguous(), length)
        idxs = self.distributed_concat(idxs.contiguous(), length)
        print('###########EXT END TEST RANK', dist.get_rank(), length)
        return edict(
            {
                "embed1": embed1,
                "idxs": idxs,
                "gts": gts,
                "p1": p1,
                "logits1": logits1,
            }
        )
    def distributed_concat(
            self,
            tensor,
            num_total_examples,):
        # output_tensors = [torch.zeros_like(tensor) for _ in range(self.trainer.world_size)]
        # torch.distributed.all_gather(output_tensors, tensor)
        # concat = torch.cat(output_tensors, dim=0)
        
        #dist.barrier()
       
        concat = tensor
        return concat[:num_total_examples]