import tensorflow as tf     # for some reason this is needed to avoid a crash
from ldm.models.autoencoder import AutoencoderKL
from omegaconf import OmegaConf
import argparse
import torchvision as tv
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from os import listdir, path
import datetime
from datasets import AdjustedMNIST, AdjustedCIFAR10, AdjustedFashionMNIST
from callbacks import ImageLogger, CUDACallback, SetupCallback


if __name__ == "__main__":
    config = OmegaConf.load("configs/autoencoder_fashionmnist.yaml")

    lightning_config = config.pop("lightning", OmegaConf.create())

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["accelerator"] = "ddp"
    trainer_config["gpus"] = 1

    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    config.model.params["image_key"] = 0

    train_ds = AdjustedFashionMNIST(train=True)
    test_ds = AdjustedFashionMNIST(train=False)
    train_ds, validation_ds = torch.utils.data.random_split(train_ds, [len(train_ds)-len(test_ds), len(test_ds)])

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=96, shuffle=True, num_workers=0, drop_last=True)
    valid_dl = torch.utils.data.DataLoader(validation_ds, batch_size=96, shuffle=False, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=96, shuffle=False, num_workers=0)


    # files = listdir(f"logs/Autoencoder_2023-03-18T21-40-57/checkpoints")
    # config.model.params["ckpt_path"] = f"logs/Autoencoder_2023-03-18T21-40-57/checkpoints/last.ckpt"

    model = AutoencoderKL(**config.model.get("params", dict()))
    model.learning_rate = config.model.base_learning_rate

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = "Autoencoder_" + now
    logdir = path.join("logs", nowname)
    ckptdir = path.join(logdir, "checkpoints")
    cfgdir = path.join(logdir, "configs")

    trainer_kwargs = dict()

    trainer_kwargs["logger"] = pl.loggers.WandbLogger(name=nowname, id=nowname)

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
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
        default_modelckpt_cfg["params"]["save_top_k"] = 3

    trainer_kwargs["checkpoint_callback"] = True

    trainer_kwargs["callbacks"] = [
        pl.callbacks.ModelCheckpoint(**default_modelckpt_cfg["params"]),
        SetupCallback(resume=False, now=now, logdir=logdir, ckptdir=ckptdir, cfgdir=cfgdir, config=config, lightning_config=lightning_config),
        ImageLogger(batch_frequency=750, max_images=8, clamp=True, increase_log_steps=False),
        CUDACallback()
    ]

    trainer = pl.Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir  ###

    try:
        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
    except KeyboardInterrupt:
        pass
