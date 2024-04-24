import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange
import kornia as K
from .ssl_joint_diffusion import SSLJointLatentDiffusion, SSLJointLatentDiffusionV2
from .joint_latent_diffusion import JointLatentDiffusion, JointLatentDiffusionAttention
from .joint_latent_diffusion_multilabel import JointLatentDiffusionMultilabel
from ..representation_transformer import RepresentationTransformer
from ..adjusted_unet import AdjustedUNet
from ..ddim import DDIMSamplerGradGuided
from ..utils import FixMatchEma, interleave, de_interleave
from sklearn.metrics import accuracy_score
from easydict import EasyDict as edict
from tqdm import tqdm
import time
from types import SimpleNamespace 
import torch.distributed as dist
from models.pl_utils import PLUL
from torch.nn import functional as F


class LatentSSLPoolingMultilabelACPL(JointLatentDiffusionMultilabel):
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 classifier_in_features,
                 classifier_hidden,
                 num_classes,
                 classification_start=0,
                 dropout=0,
                 classification_loss_weight=1.0,
                 classification_key=1,
                 augmentations = True,
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
            classification_start=classification_start,
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
        self.acpl_loop=0

    def get_sampl(self):
        print("sampling_method, gradient_guided_samplings", self.sampling_method, self.gradient_guided_sampling)
        
    def get_input(self,
                  batch,
                  k,
                  return_first_stage_outputs=False,
                  force_c_encode=False,
                  cond_key=None,
                  return_original_cond=False,
                  bs=None):
        if self.training is True:
            batch = batch[1][0]
        return super().get_input(
            batch,
            k,
            return_first_stage_outputs,
            force_c_encode,
            cond_key,
            return_original_cond,
            bs
        )

    def get_train_classification_input(self, batch, k):
        if type(batch[0])==list:
            x = batch[0][k]
        else:
            x = batch[k]
        x = self.to_latent(x, arrange=True)
        y = batch[self.classification_key]
        return x, y 

    def get_valid_classification_input(self, batch, k):
        x = batch[k]
        x = self.to_latent(x, arrange=True)
        y = batch[self.classification_key]
        return x, y
    
    def get_input(self,
                  batch,
                  k,
                  return_first_stage_outputs=False,
                  force_c_encode=False,
                  cond_key=None,
                  return_original_cond=False,
                  bs=None):
        if type(batch[0])==list:
            batch = batch[0]
        # k=0 should mean img for tuple (img, label). 
        #Here it means sth different to have mathcing idx: (img, img_weak, img_strong)
        return super().get_input(
            batch,
            k,
            return_first_stage_outputs,
            force_c_encode,
            cond_key,
            return_original_cond,
            bs
        )

    def training_step(self, batch, batch_idx):
        # old version:
        # if self.global_step%4!=0:
        #     cat_x = torch.cat((batch[0][0], batch[1][0]))
        #     cat_y = torch.cat((batch[0][1], batch[1][1]))
        #     loss, loss_dict = self.shared_step((cat_x,cat_y))
        # else:
        #     loss, loss_dict = self.shared_step(batch[1])

        #non-overfitting version:
        loss, loss_dict = self.shared_step(batch[1])
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        loss = self.train_classification_step(batch[0], loss)
        return loss

    def train_classification_step(self, batch, loss):
        if self.classification_start > self.global_step:
            return loss
        
        if self.global_step%4!=0:
            return loss
        
        loss_dict = {}

        x, y = self.get_train_classification_input(batch, self.first_stage_key)
        t = torch.zeros((x.shape[0],), device=self.device).long()
            
        loss_classification, accuracy, y_pred = self.do_classification(x, t, y)
        #self.auroc_train.update(y_pred[:,:self.used_n_classes], y[:,:self.used_n_classes])

        loss += loss_classification * self.classification_loss_weight

        #self.log('train/auroc', self.auroc_train, on_step=False, on_epoch=True)
        loss_dict.update({'train/loss_classification': loss_classification})
        loss_dict.update({'train/loss_full': loss})
        loss_dict.update({'train/accuracy': accuracy})
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        return loss
    
    def forward_with_repr(self, inputs):
        unet: AdjustedUNet = self.model.diffusion_model
        inputs = self.to_latent(inputs, arrange=True)
        t = torch.zeros((inputs.shape[0],), device=self.device).long()
        representations = unet.just_representations(inputs, t, pooled=False)
        representations = self.transform_representations(representations)
        y_pred = self.classifier(representations)
        return y_pred, F.normalize(representations, dim=-1, p=2)
    
    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        return super().log_images(batch=batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs)

    def on_train_epoch_end(self) -> None:
        # we have psudolabel-enhanced trainings in between epoch ends.
        # That;s why we have strange if-conditions and order

        # here we deal with after-loop operations first. First we have to close epoch 29.
        if self.current_epoch==1\
            or self.current_epoch==2:
            if dist.get_rank()==0:
                self.acpl_actions_after_training_loop()

        # here we prepare operations for next loop. At the end of 29th epoch we prepare for 30th epoch.
        if self.current_epoch==0 \
            or self.current_epoch==1:    
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