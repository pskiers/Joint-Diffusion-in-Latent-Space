from dataclasses import dataclass
from omegaconf import OmegaConf
import argparse
import torch
import pytorch_lightning as pl
from models import get_model_class
from dataloading import get_dataloaders
from os import path, environ
from pathlib import Path
import datetime
from callbacks import ImageLogger, CUDACallback, SetupCallback, FIDScoreLogger


if __name__ == "__main__":
    environ["WANDB__SERVICE_WAIT"] = "300"

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=Path, required=True, help="path to config file")
    parser.add_argument("--checkpoint", "-c", type=Path, required=False, help="path to model checkpoint file")
    args = parser.parse_args()
    config_path = str(args.path)
    checkpoint_path = str(args.checkpoint) if args.checkpoint is not None else None

    config = OmegaConf.load(config_path)
    # config = OmegaConf.load(
    #     "configs/baselines/class_conditioned_ddpm/cifar10.yaml"
    # )

    lightning_config = config.pop("lightning", OmegaConf.create())

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["devices"] = -1
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    dl_config_orig = config.pop("dataloaders")
    dl_config = OmegaConf.to_container(dl_config_orig, resolve=True)
    train_dls, test_dl = get_dataloaders(dl_config)

    if checkpoint_path is not None:
        config.model.params["ckpt_path"] = checkpoint_path
    # config.model.params["ckpt_path"] = f"logs/JointDiffusionAttention_2023-10-08T22-42-51/checkpoints/last.ckpt"

    model = get_model_class(config.model.get("model_type"))(
        **config.model.get("params", dict())
    )

    # model.supervised_dataloader = train_dl_supervised

    model.learning_rate = config.model.base_learning_rate

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = model.__class__.__name__ + "_" + now
    logdir = path.join("logs", nowname)
    ckptdir = path.join(logdir, "checkpoints")
    cfgdir = path.join(logdir, "configs")

    trainer_kwargs = dict()
    try:
        per_class = f'{dl_config["train"][0]["ssl_split"]["num_labeled"]} per class'
    except Exception:
        per_class = "all labels"
    tags = [
        dl_config["validation"]["name"],
        per_class,
        config.model.get("model_type"),
    ]
    trainer_kwargs["logger"] = pl.loggers.WandbLogger(
        name=nowname, id=nowname, tags=tags
    )

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
            lightning_config=lightning_config,
            dl_config=dl_config_orig,
        ),
        CUDACallback(),
    ]
    if (img_logger_cfg := callback_cfg.get("img_logger", None)) is not None:
        trainer_kwargs["callbacks"].append(ImageLogger(**img_logger_cfg))

    if (fid_cfg := callback_cfg.get("fid_logger", None)) is not None:
        fid_cfg = dict(fid_cfg)
        fid_cfg["real_dl"] = (
            train_dls[1] if type(train_dls) in (tuple, list) else train_dls
        )  # first dataloader should contain the original images
        fid_cfg["device"] = torch.device("cuda")
        trainer_kwargs["callbacks"].append(FIDScoreLogger(**fid_cfg))

    trainer = pl.Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

    trainer.fit(
        model,
        train_dataloaders=train_dls,
        val_dataloaders=test_dl,
    )
