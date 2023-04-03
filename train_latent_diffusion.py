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
from datasets import AdjustedMNIST, AdjustedCIFAR10
from callbacks import ImageLogger, CUDACallback, SetupCallback, FIDScoreLogger


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
    train_ds, validation_ds = torch.utils.data.random_split(train_ds, [len(train_ds)-len(test_ds), len(test_ds)], generator=torch.Generator().manual_seed(42))

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
    valid_dl = torch.utils.data.DataLoader(validation_ds, batch_size=128, shuffle=False, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    # i = 23
    # files = listdir(f"lightning_logs/version_{i}/checkpoints")
    config.model.params["ckpt_path"] = f"logs/LatentDiffusion_2023-03-28T08-23-58/checkpoints/last.ckpt"

    model = LatentDiffusion(**config.model.get("params", dict()))
    model.learning_rate = config.model.base_learning_rate

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = "LatentDiffusion_" + now
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
        ImageLogger(batch_frequency=2000, max_images=8, clamp=True, increase_log_steps=False, log_images_kwargs={"inpaint": False}),
        FIDScoreLogger(batch_frequency=400, samples_amount=5000, metrics_batch_size=64, device=torch.device("cuda"), means_path="fid_saves/cifar10/cifar_means.npy", sigma_path="fid_saves/cifar10/cifar_sigmas.npy"),
        CUDACallback()
    ]

    trainer = pl.Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)

#     real_img, _ = test_ds[0]
#     print(real_img.max(), real_img.min())

#     imgs = model.sample(None, batch_size=8)
#     imgs = model.decode_first_stage(imgs)
#     print(imgs.max(), imgs.min())
#     grid = tv.utils.make_grid(imgs)
#     plt.imshow(grid.permute(1, 2, 0))
#     plt.show()
