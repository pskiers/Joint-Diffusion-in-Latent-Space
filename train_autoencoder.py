import tensorflow as tf     # for some reason this is needed to avoid a crash
from ldm.models.autoencoder import AutoencoderKL
from omegaconf import OmegaConf
import argparse
import torchvision as tv
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
# from os import listdir
from datasets.mnist import AdjustedMNIST


if __name__ == "__main__":
    config = OmegaConf.load("configs/autoencoder_mnist.yaml")

    lightning_config = config.pop("lightning", OmegaConf.create())

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["accelerator"] = "ddp"
    trainer_config["gpus"] = 1

    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    config.model.params["image_key"] = 0

    train_ds = AdjustedMNIST(train=True)
    test_ds = AdjustedMNIST(train=False)
    train_ds, validation_ds = torch.utils.data.random_split(train_ds, [len(train_ds)-len(test_ds), len(test_ds)])

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
    valid_dl = torch.utils.data.DataLoader(validation_ds, batch_size=256, shuffle=False, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    trainer = pl.Trainer.from_argparse_args(trainer_opt)

    # i = 23
    # files = listdir(f"lightning_logs/version_{i}/checkpoints")
    # config.model.params["ckpt_path"] = f"lightning_logs/version_{i}/checkpoints/{files[0]}"

    model = AutoencoderKL(**config.model.get("params", dict()))
    model.learning_rate = config.model.base_learning_rate

    try:
        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
    except KeyboardInterrupt:
        pass

    for img, _ in test_dl:
        dec, posterior = model(img.transpose(1, 3))
        imgs = torch.concat([img.transpose(1, 3), dec.detach()], dim=0)
        grid = tv.utils.make_grid(imgs)
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()
        break
