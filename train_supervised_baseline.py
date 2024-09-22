import datetime
from os import path
from omegaconf import OmegaConf
import argparse
import torch
import pytorch_lightning as pl
from dataloading import AdjustedSVHN
from models import WideResNet, WideResNetEncoder, DDPM_Wide_ResNet
from callbacks import CUDACallback, SetupCallback


if __name__ == "__main__":
    train_ds = AdjustedSVHN(train="train")
    test_ds = AdjustedSVHN(train="test")

    train_ds, validation_ds = torch.utils.data.random_split(train_ds, [len(train_ds)-len(test_ds), len(test_ds)], generator=torch.Generator().manual_seed(42))

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    valid_dl = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    model = DDPM_Wide_ResNet()
    model.lr = 0.001

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

    trainer_kwargs["callbacks"] = [
        pl.callbacks.ModelCheckpoint(**default_modelckpt_cfg["params"]),
        SetupCallback(resume=False, now=now, logdir=logdir, ckptdir=ckptdir, cfgdir=cfgdir, config={}, lightning_config={}),
        CUDACallback()
    ]

    trainer_config = OmegaConf.create()
    trainer_config["accelerator"] = "ddp"
    trainer_config["devices"] = -1
    trainer_opt = argparse.Namespace(**trainer_config)

    trainer = pl.Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
