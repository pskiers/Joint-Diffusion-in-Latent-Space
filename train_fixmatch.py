from omegaconf import OmegaConf
import argparse
import pytorch_lightning as pl
from models import *
from datasets import *
from os import path
import datetime
from callbacks import CUDACallback, SetupCallback
from dataclasses import dataclass
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


@dataclass
class Args:
    num_labeled: int = 40
    num_classes: int = 10
    expand_labels: bool = True
    batch_size: int = 64
    eval_step: int = 1024


if __name__ == "__main__":
    config = OmegaConf.load("configs/cifar10-joint-standard-diffusion.yaml") # not really important

    lightning_config = config.pop("lightning", OmegaConf.create())

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # trainer_config["accelerator"] = "ddp"
    trainer_config["gpus"] = 1
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    args = Args()
    train_sampler = RandomSampler
    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS["cifar10_multi"](args, './data')
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=60,
        num_workers=4,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=42,
        num_workers=4,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=64,
        num_workers=4)


    model = MeanMatch(batch_size=60)

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = model.__class__.__name__ + "_" + now
    logdir = path.join("logs", nowname)
    ckptdir = path.join(logdir, "checkpoints")
    cfgdir = path.join(logdir, "configs")

    trainer_kwargs = dict()

    trainer_kwargs["logger"] = pl.loggers.WandbLogger(name=nowname, id=nowname)

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

    trainer_kwargs["checkpoint_callback"] = True
    trainer_kwargs["max_steps"] = 2**20

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

    trainer = pl.Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir

    trainer.fit(
        model,
        train_dataloaders=[labeled_trainloader, unlabeled_trainloader],
        val_dataloaders=test_loader,
    )