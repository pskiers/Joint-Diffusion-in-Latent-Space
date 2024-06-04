from dataclasses import dataclass
from omegaconf import OmegaConf
import argparse
import torch
import pytorch_lightning as pl
import torch.distributed
from models import get_model_class
from datasets import get_dataloaders
from os import path, environ
from pathlib import Path
import datetime
from callbacks import ImageLogger, CUDACallback, SetupCallback, FIDScoreLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import wandb
from datasets import ACPLDataModule
from models import MultilabelClassifierACPL
import os

if __name__ == "__main__":
    environ["WANDB__SERVICE_WAIT"] = "300"

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=Path, required=True, help="path to config file")
    parser.add_argument("--checkpoint", "-c", type=Path, required=False, help="path to model checkpoint file")
    parser.add_argument("--prefix", "-pref", type=str, required=False, help="prefix to experiment")
    args = parser.parse_args()
    config_path = str(args.path)
    checkpoint_path = str(args.checkpoint) if args.checkpoint is not None else None

    config = OmegaConf.load(config_path)
    # config = OmegaConf.load("configs/standard_diffusion/semi-supervised/diffmatch_wide_resnet_unet/25_per_class/svhn.yaml")

    lightning_config = config.pop("lightning", OmegaConf.create())

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    if checkpoint_path is not None:
        config.model.params["ckpt_path"] = checkpoint_path

    model = get_model_class(config.model.get("model_type"))(**config.model.get("params", dict()))
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = args.prefix + "_" + model.__class__.__name__ + "_" + now
    trainer_kwargs = dict()
    
    dl_config = config.pop("dataloaders")
    tags = [
        dl_config["name"],
        "all labels" if dl_config["num_labeled"] is None else f"{dl_config['num_labeled']} per class",
        config.model.get("model_type")
    ]

    logdir = path.join(config.model.logdir,"logs", nowname)
    ckptdir = path.join(logdir, "checkpoints")
    cfgdir = path.join(logdir, "configs")
    trainer_kwargs["logger"] = pl.loggers.WandbLogger(name=nowname, tags=tags, project = "Joint-Diffusion-in-Latent-Space", resume="allow")#,id=nowname)

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
        }
    }

    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 1
        default_modelckpt_cfg["params"]["mode"] = 'max' if 'auroc'in model.monitor or 'acc' in model.monitor else 'min'

    callback_cfg = lightning_config.get("callbacks", OmegaConf.create())
    trainer_kwargs["callbacks"] = [
        pl.callbacks.ModelCheckpoint(**default_modelckpt_cfg["params"]),
        SetupCallback(
            resume=True,
            now=now,
            logdir=logdir,
            ckptdir=ckptdir,
            cfgdir=cfgdir,
            config=config,
            lightning_config=lightning_config
        ),
        CUDACallback(),
        LearningRateMonitor(logging_interval='step')
    ]

    trainer = pl.Trainer.from_argparse_args(trainer_opt, check_val_every_n_epoch=1, 
        reload_dataloaders_every_n_epochs = 1, **trainer_kwargs)
    
   
    trainer.logdir = logdir
    labeled_items, test_items, anchor_items, unlabeled_items, acpl_loader = get_dataloaders(**dl_config)
    
    acpl_data_datamodule = ACPLDataModule(train_loader=labeled_items[0], #get only dataloader
                                          val_loader=test_items[0], #get only dataloader
                                        #   anchor_loader=anchor_items[0],
                                        #   unlabeled_loader=unlabeled_items[0]
                                        )
    
    if os.environ.get('LOCAL_RANK', 0)==0:
        trainer.labeled_loader, trainer.labeled_dataset, trainer.labeled_sampler = labeled_items 
        trainer.test_loader, trainer.test_dataset, trainer.test_sampler = test_items 
        trainer.anchor_loader, trainer.anchor_dataset, trainer.anchor_sampler = anchor_items 
        trainer.unlabeled_loader, trainer.unlabeled_dataset, trainer.unlabeled_sampler = unlabeled_items 
        trainer.last_loop_anchor_len = len(trainer.anchor_dataset)
        trainer.acpl_loader = acpl_loader
    
    trainer.fit(
        model,
        acpl_data_datamodule
    )
    
    # print('HERREEEEEEEEERRRREEEE we check how many times script runs')
    # #resume
    # trainer_kwargs["resume_from_checkpoint"] = ckptdir+'/last.ckpt'
    # trainer_kwargs["max_epochs"]=20
    # acpl_data_datamodule = ACPLDataModule(train_loader=trainer.labeled_loader, 
    #                                       val_loader=trainer.test_loader,
    #                                     #   anchor_loader=trainer.anchor_loader,
    #                                     #   unlabeled_loader=trainer.unlabeled_loader
    #                                     )
    # trainer = pl.Trainer.from_argparse_args(trainer_opt, check_val_every_n_epoch=5, reload_dataloaders_every_n_epochs = 500, **trainer_kwargs)
    # trainer.fit(model, acpl_data_datamodule)

###https://lightning.ai/forums/t/modifying-the-trainer-when-calling-trainer-fit-multiple-times/1743
###https://stackoverflow.com/questions/73154831/no-predict-dataloader-method-defined-to-run-trainer-predict
### https://github.com/Lightning-AI/pytorch-lightning/discussions/16258
### https://github.com/Lightning-AI/pytorch-lightning/discussions/13041
### https://lightning.ai/forums/t/validation-sanity-check-hangs-after-all-gather/1469