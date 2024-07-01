from dataclasses import dataclass
from omegaconf import OmegaConf
import argparse
import torch
import pytorch_lightning as pl
from models import get_model_class
from datasets import get_dataloaders
from os import path, environ
from pathlib import Path
import datetime
from callbacks import ImageLogger, CUDACallback, SetupCallback, FIDScoreLogger
from datasets import ACPLDataModule
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

    lightning_config = config.pop("lightning", OmegaConf.create())

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_opt = argparse.Namespace(**trainer_config)
    # if "resume_from_checkpoint" not in trainer_opt:
    #     raise NotImplemented("ACPL HAS TO BE RESUMED")
    lightning_config.trainer = trainer_config

    dl_config = config.pop("dataloaders")
    diffusion_loader, labeled_items, test_items, anchor_items, unlabeled_items, acpl_loader = get_dataloaders(**dl_config)

    if checkpoint_path is not None:
        config.model.params["ckpt_path"] = checkpoint_path

    model = get_model_class(config.model.get("model_type"))(**config.model.get("params", dict()))

    model.learning_rate = config.model.base_learning_rate
    try:
        model.classifier_lr = config.model.classifier_lr
        print(['CLASSIFIER HAS SEPARATE LR'])
    except:
        model.classifier_lr=None
        print(['CLASSIFIER HAS THE SAME LR'])

    model.sampling_method = "unconditional"
    model.gradient_guided_sampling=False

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = args.prefix + "_" + model.__class__.__name__ + "_" + now
    logdir = path.join(config.model.logdir,"logs", nowname)
    ckptdir = path.join(logdir, "checkpoints")
    cfgdir = path.join(logdir, "configs")

    trainer_kwargs = dict()
    tags = [
        dl_config["name"],
        "all labels" if dl_config["num_labeled"] is None else f"{dl_config['num_labeled']} per class",
        config.model.get("model_type")
    ]
    trainer_kwargs["logger"] = pl.loggers.WandbLogger(name=nowname, id=nowname, tags=tags, project = "Joint-Diffusion-in-Latent-Space")

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
            resume=False,
            now=now,
            logdir=logdir,
            ckptdir=ckptdir,
            cfgdir=cfgdir,
            config=config,
            lightning_config=lightning_config
        ),
        CUDACallback()
    ]
    if (img_logger_cfg := callback_cfg.get("img_logger", None)) is not None:
        trainer_kwargs["callbacks"].append(ImageLogger(**img_logger_cfg))

    if (fid_cfg := callback_cfg.get("fid_logger", None)) is not None:
        pass
        # fid_cfg = dict(fid_cfg)
        # fid_cfg["real_dl"] = train_dls[1] if type(train_dls) in (tuple, list) else train_dls
        # fid_cfg["device"] = torch.device("cuda")
        # trainer_kwargs["callbacks"].append(FIDScoreLogger(**fid_cfg))

    trainer = pl.Trainer.from_argparse_args(trainer_opt, **trainer_kwargs, 
                                            reload_dataloaders_every_n_epochs = 1)
    trainer.logdir = logdir
    
    print("[WARNING] train_joint_diff_acpl.py - IN ACPL datamodule TO avoid errors we have val loader mocked with train data 2%!!!!")
    acpl_data_datamodule = ACPLDataModule(train_loader=[labeled_items[0], diffusion_loader], #get only dataloader
                                          val_loader=test_items[0] #get only dataloader
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