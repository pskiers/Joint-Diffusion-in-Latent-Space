import tensorflow as tf     # for some reason this is needed to avoid a crash
from ldm.models.diffusion.ddpm import LatentDiffusion
from omegaconf import OmegaConf
import argparse
import torchvision as tv
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from os import listdir, path
import datetime
from datasets.mnist import AdjustedMNIST
from datasets.cifar10 import AdjustedCIFAR10
from callbacks import ImageLogger, CUDACallback, SetupCallback


if __name__ == "__main__":
    config = OmegaConf.load("configs/cifar10-ldm-autoencoder_cifar10.yaml")

    lightning_config = config.pop("lightning", OmegaConf.create())

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["accelerator"] = "ddp"
    trainer_config["gpus"] = 1
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    train_ds = AdjustedCIFAR10(train=True)
    test_ds = AdjustedCIFAR10(train=False)
    train_ds, validation_ds = torch.utils.data.random_split(train_ds, [len(train_ds)-len(test_ds), len(test_ds)])

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
    valid_dl = torch.utils.data.DataLoader(validation_ds, batch_size=256, shuffle=False, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    # i = 23
    # files = listdir(f"lightning_logs/version_{i}/checkpoints")
    # config.model.params["ckpt_path"] = f"lightning_logs/version_{i}/checkpoints/{files[0]}"

    model = LatentDiffusion(**config.model.get("params", dict()))
    model.learning_rate = config.model.base_learning_rate

    nowname = now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logdir = path.join("logs", nowname)
    ckptdir = path.join(logdir, "checkpoints")
    cfgdir = path.join(logdir, "configs")

    trainer_kwargs = dict()

    trainer_kwargs["logger"] = pl.loggers.TestTubeLogger(name="testtube", save_dir=logdir)

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
        ImageLogger(batch_frequency=750, max_images=4, clamp=True, increase_log_steps=False),
        CUDACallback()
    ]

    trainer = pl.Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir  ###

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)

    imgs = model.sample(None, batch_size=8)
    imgs = model.decode_first_stage(imgs)
    grid = tv.utils.make_grid(imgs)
    plt.imshow(grid.permute(2, 1, 0))
    plt.show()
