# import tensorflow as tf
from omegaconf import OmegaConf
import argparse
import torch
import pytorch_lightning as pl
from models import JointLatentDiffusionNoisyClassifier, JointLatentDiffusion, SSLJointDiffusion, SSLJointDiffusionV2, DiffMatch, DiffMatchV2
from datasets import AdjustedMNIST, AdjustedSVHN, AdjustedSVHN, GTSRB, equal_labels_random_split
from os import listdir, path
import datetime
from callbacks import ImageLogger, CUDACallback, SetupCallback, FIDScoreLogger
import matplotlib.pyplot as plt
import torchvision as tv


if __name__ == "__main__":
    config = OmegaConf.load("configs/svhn-joint-diffusion.yaml")

    lightning_config = config.pop("lightning", OmegaConf.create())

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["accelerator"] = "ddp"
    trainer_config["gpus"] = 1
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    train_ds = AdjustedSVHN(train="train")
    test_ds = AdjustedSVHN(train="test")

    # train_ds, validation_ds = torch.utils.data.random_split(train_ds, [len(train_ds)-int(0.2*len(train_ds)), int(0.2*len(train_ds))], generator=torch.Generator().manual_seed(42))
    train_ds, validation_ds = torch.utils.data.random_split(train_ds, [len(train_ds)-len(test_ds), len(test_ds)], generator=torch.Generator().manual_seed(42))
    _, train_supervised = equal_labels_random_split(train_ds, labels=[i for i in range(10)], amount_per_class=100, generator=torch.Generator().manual_seed(42))

    train_dl_unsupervised = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
    train_dl_supervised = torch.utils.data.DataLoader(train_supervised, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    valid_dl = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    # test_dl = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    # config.model.params["ckpt_path"] = f"logs/JointDiffusion_2023-05-16T02-41-02/checkpoints/last.ckpt"

    model = DiffMatchV2(**config.model.get("params", dict()))
    # model.min_confidence = 0
    model.supervised_skip_n = 0
    model.supervised_skip_current = 0
    model.supervised_dataloader = train_dl_supervised

    model.learning_rate = config.model.base_learning_rate

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
        SetupCallback(resume=False, now=now, logdir=logdir, ckptdir=ckptdir, cfgdir=cfgdir, config=config, lightning_config=lightning_config),
        ImageLogger(batch_frequency=2000, max_images=10, clamp=True, increase_log_steps=False, log_images_kwargs={"N": 10, "inpaint": False, "sample_classes": [0,1,2,3,4,5,6,7,8,9]}),
        # FIDScoreLogger(batch_frequency=10000, samples_amount=10000, metrics_batch_size=64, device=torch.device("cuda")),
        CUDACallback()
    ]

    trainer = pl.Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir

    trainer.fit(model, train_dataloaders=train_dl_unsupervised, val_dataloaders=valid_dl)
    # model.sample_classes=torch.tensor([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9])
    # model.sample_grad_scale = 60
    # samples = model.sample(cond=None, batch_size=20)
    # samples = model.decode_first_stage(samples.to(model.device))
    # grid = tv.utils.make_grid(samples.cpu())
    # plt.imshow(grid.permute(1,2,0))
    # plt.show()
