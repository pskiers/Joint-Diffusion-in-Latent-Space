from typing import Any
import torch
import torch.nn as nn
import torchvision as tv
import pytorch_lightning as pl


class ResNetWithKD(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        old_model,
        new_model,
        old_classes,
        new_classes,
        *args: Any,
        head_in=1000,
        head_hidden=512,
        lr=0.001,
        kd_classification_weight=1.0,
        classification_weight=0.01,
        ckpt_path=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conv = tv.models.resnet18()
        self.relu = nn.ReLU()
        self.head_in = nn.Linear(head_in, head_hidden)
        self.head_out = nn.Linear(head_hidden, num_classes)
        self.lr = lr

        self.kd_models = {"old_model": old_model, "new_model": new_model}  # in dict so that pytorch does not see this via parameters()
        self.old_classes = torch.tensor(old_classes, device=self.device)
        self.new_classes = torch.tensor(new_classes, device=self.device)

        self.kd_classification_weight = kd_classification_weight
        self.classification_weight = classification_weight

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)
        out = self.head_in(out)
        out = self.relu(out)
        return self.head_out(out)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs = imgs.transpose(1, 3)
        preds = self(imgs)

        # classification
        loss = 0.0
        if self.classification_weight != 0:
            loss_classification = nn.functional.cross_entropy(preds, labels)
            loss += self.classification_weight * loss_classification
            acc = torch.sum(torch.argmax(preds, dim=1) == labels) / len(labels)
            loss_dict = {"train/loss_classification": loss_classification, "train/accuracy": acc}
        # knowledge distillation
        if self.kd_classification_weight != 0:
            old_classes_mask = torch.any(labels.unsqueeze(-1) == self.old_classes.to(self.device), dim=-1)
            new_classes_mask = torch.any(labels.unsqueeze(-1) == self.new_classes.to(self.device), dim=-1)
            
            loss_old = 0
            if old_classes_mask.sum() != 0:
                with torch.no_grad():
                    old_preds = self.kd_models["old_model"](imgs[old_classes_mask])
                    old_preds = nn.functional.softmax(old_preds, dim=1).detach()
                loss_old = nn.functional.cross_entropy(preds[old_classes_mask], old_preds)

            loss_new = 0
            if new_classes_mask.sum() != 0:
                with torch.no_grad():
                    new_preds = self.kd_models["new_model"](imgs[new_classes_mask])
                    new_preds = nn.functional.softmax(new_preds, dim=1).detach()
                loss_new = nn.functional.cross_entropy(preds[new_classes_mask], new_preds)
            loss += self.kd_classification_weight * (loss_new + loss_old)

            loss_dict.update({f'train/loss_classifier_kd': loss_old + loss_new})

        loss_dict.update({f'train/loss': loss})
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs = imgs.transpose(1, 3)
        preds = self(imgs)

        # classification
        loss = 0.0
        if self.classification_weight != 0:
            loss_classification = nn.functional.cross_entropy(preds, labels)
            loss += self.classification_weight * loss_classification
            acc = torch.sum(torch.argmax(preds, dim=1) == labels) / len(labels)
            loss_dict = {"val/loss_classification": loss_classification, "val/accuracy": acc}
        # knowledge distillation
        if self.kd_classification_weight != 0:
            old_classes_mask = torch.any(labels.unsqueeze(-1) == self.old_classes.to(self.device), dim=-1)
            new_classes_mask = torch.any(labels.unsqueeze(-1) == self.new_classes.to(self.device), dim=-1)
            
            loss_old = 0
            if old_classes_mask.sum() != 0:
                with torch.no_grad():
                    old_preds = self.kd_models["old_model"](imgs[old_classes_mask])
                    old_preds = nn.functional.softmax(old_preds, dim=1).detach()
                loss_old = nn.functional.cross_entropy(preds[old_classes_mask], old_preds)

            loss_new = 0
            if new_classes_mask.sum() != 0:
                with torch.no_grad():
                    new_preds = self.kd_models["new_model"](imgs[new_classes_mask])
                    new_preds = nn.functional.softmax(new_preds, dim=1).detach()
                loss_new = nn.functional.cross_entropy(preds[new_classes_mask], new_preds)
            loss += self.kd_classification_weight * (loss_new + loss_old)

            loss_dict.update({f'val/loss_classifier_kd': loss_old + loss_new})
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=False, on_epoch=False)
        return loss_dict

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
