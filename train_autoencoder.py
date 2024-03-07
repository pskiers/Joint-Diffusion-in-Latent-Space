from ldm.models.autoencoder import AutoencoderKL, VQModel
from omegaconf import OmegaConf
import argparse
import pytorch_lightning as pl
from datasets import get_dataloaders
from os import path, environ
from pathlib import Path
import datetime
from callbacks import ImageLogger, CUDACallback, SetupCallback


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
    lightning_config.trainer = trainer_config

    dl_config = config.pop("dataloaders")
    train_dls, val_dl, test_dl = get_dataloaders(**dl_config)

    if checkpoint_path is not None:
        config.model.params["ckpt_path"] = checkpoint_path

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    if 'AutoencoderKL' in config.model.target:
        model = AutoencoderKL(**config.model.get("params", dict()))
        nowname = args.prefix + "_AutoencoderKL_" + now
    elif 'VQModel' in config.model.target:
        model = VQModel(**config.model.get("params", dict()))
        nowname = args.prefix + "_AutoencoderVQ_" + now
    else:
        raise NotImplementedError
    
    model.learning_rate = config.model.base_learning_rate

    logdir = path.join(config.logdir, "logs", nowname)
    ckptdir = path.join(logdir, "checkpoints")
    cfgdir = path.join(logdir, "configs")

    trainer_kwargs = dict()
    tags = [
        dl_config["name"],
        "all labels" if dl_config["num_labeled"] is None else f"{dl_config['num_labeled']} per class",
        config.model.get("model_type")
    ]
    trainer_kwargs["logger"] = pl.loggers.WandbLogger(name=nowname, id=nowname, project = "Joint-Diffusion-in-Latent-Space")

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
            lightning_config=lightning_config
        ),
        CUDACallback()
    ]
    if (img_logger_cfg := callback_cfg.get("img_logger", None)) is not None:
        trainer_kwargs["callbacks"].append(ImageLogger(**img_logger_cfg))

    trainer = pl.Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir

    trainer.fit(
        model=model,
        train_dataloaders=train_dls,
        val_dataloaders=test_dl
    )
