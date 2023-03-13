import tensorflow as tf
from ldm.models.diffusion.ddpm import LatentDiffusion
from omegaconf import OmegaConf
import argparse
import torch
import pytorch_lightning as pl
from os import listdir
from models.ClassifierOnLatentDiffusion import ClassifierOnLatentDiffusion
from datasets.mnist import AdjustedMNIST


if __name__ == "__main__":
    config = OmegaConf.load("configs/mnist-ldm-autoencoder_mnist.yaml")

    lightning_config = config.pop("lightning", OmegaConf.create())

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["accelerator"] = "ddp"
    trainer_config["gpus"] = 1
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    train_ds = AdjustedMNIST(train=True)
    test_ds = AdjustedMNIST(train=False)
    train_ds, validation_ds = torch.utils.data.random_split(train_ds, [len(train_ds)-len(test_ds), len(test_ds)])

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
    valid_dl = torch.utils.data.DataLoader(validation_ds, batch_size=256, shuffle=False, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    trainer = pl.Trainer.from_argparse_args(trainer_opt)

    i = 23
    files = listdir(f"lightning_logs/version_{i}/checkpoints")
    config.model.params["ckpt_path"] = f"lightning_logs/version_{i}/checkpoints/{files[0]}"

    model = LatentDiffusion(**config.model.get("params", dict()))
    model.learning_rate = config.model.base_learning_rate

    classifier_model = ClassifierOnLatentDiffusion(model, 10, torch.device("cuda"))

    device = torch.device("cuda")
    criterion = torch.nn.functional.cross_entropy
    optim = torch.optim.Adam(classifier_model.parameters(), lr=0.001)
    epochs = 100

    for epoch in range(epochs):
        train_accuracies = []
        train_losses = []
        valid_accuracies = []
        valid_losses = []

        for imgs, labels in train_dl:
            imgs, labels = imgs.transpose(1, 3).to(device), labels.to(device)
            preds = classifier_model(imgs)

            loss = criterion(preds, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_losses.append(loss.item())
            train_accuracies.append(torch.sum(torch.argmax(preds, dim=1) == labels) / len(labels))

        for imgs, labels in valid_dl:
            imgs, labels = imgs.transpose(1, 3).to(device), labels.to(device)

            with torch.no_grad():
                preds = classifier_model(imgs)

            loss = criterion(preds, labels)

            valid_losses.append(loss.item())
            valid_accuracies.append(torch.sum(torch.argmax(preds, dim=1) == labels) / len(labels))

        print(f"Epoch {epoch}\t Train loss: {sum(train_losses)/len(train_losses)}\t Train acc: {sum(train_accuracies)/len(train_accuracies)}\t Validation loss: {sum(valid_losses)/len(valid_losses)}\t Validation acc: {sum(valid_accuracies)/len(valid_accuracies)}")

