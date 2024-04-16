# import tensorflow as tf
from ldm.models.diffusion.ddpm import LatentDiffusion
from omegaconf import OmegaConf
import argparse
import torch
import pytorch_lightning as pl
from models import ClassifierOnLatentDiffusion, AttentionOnLatentDiffusion, FixMatch
from dataloading import AdjustedMNIST, AdjustedSVHN, AdjustedCIFAR10
from os import listdir, path
import datetime
from callbacks import ImageLogger, CUDACallback, SetupCallback, FIDScoreLogger

if __name__ == "__main__":
    config = OmegaConf.load("configs/svhn-simple-classifier-ldm.yaml")

    lightning_config = config.pop("lightning", OmegaConf.create())

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["accelerator"] = "ddp"
    trainer_config["gpus"] = 1
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    train_ds = AdjustedCIFAR10(train="train")
    test_ds = AdjustedCIFAR10(train="test")
    train_ds, validation_ds = torch.utils.data.random_split(train_ds, [len(train_ds)-len(test_ds), len(test_ds)], generator=torch.Generator().manual_seed(42))

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
    valid_dl = torch.utils.data.DataLoader(validation_ds, batch_size=256, shuffle=False, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    model = LatentDiffusion(**config.model.get("params", dict()))
    model.learning_rate = config.model.base_learning_rate

    classifier_model = AttentionOnLatentDiffusion(
        trained_diffusion=model,
        attention_config={
            "num_classes": 10,
            "channels": 160,
            "dim_head": 32,
            "context_dims": [160, 160, 160, 160, 320, 320, 320, 480, 480, 480],
            "mlp_size": 10240,
            "hidden_size": 1024,
        }
    )
    # classifier_model = ClassifierOnLatentDiffusion(
    #     trained_diffusion=model,
    #     num_classes=10,
    #     in_features=3040,
    #     hidden_layer=2048,)

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = classifier_model.__class__.__name__ + "_" + now
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
        SetupCallback(resume=False, now=now, logdir=logdir, ckptdir=ckptdir, cfgdir=cfgdir, config=config, lightning_config=lightning_config),
        CUDACallback()
    ]

    trainer = pl.Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir

    trainer.fit(classifier_model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
