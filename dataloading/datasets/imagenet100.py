from typing import Callable, Optional, Tuple
import numpy as np
import os
from PIL import Image
import torchvision as tv
import torch

from .base import BaseDataset, Split


class Imagenet100(BaseDataset):
    def __init__(
        self,
        root: str = "data",
        split: Split = Split.TRAIN,
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        load_preprocessed: bool = True,
        **kwargs
    ) -> None:
        super().__init__(root, split, download, transform, target_transform, **kwargs)
        self.load_preprocessed = load_preprocessed
        if load_preprocessed:
            prefix = "train" if split == Split.TRAIN else "val"
            imgs = torch.load(os.path.join(root, "imagenet100", f"{prefix}64x64_imgs"))
            targets = torch.load(os.path.join(root, "imagenet100", f"{prefix}64x64_labels"))
            self.set_data(imgs)
            self.set_targets(targets)
            return
        else:
            dataset = tv.datasets.ImageFolder(root=os.path.join(root, "imagenet100", "train" if split == Split.TRAIN else "val"))
            self.set_data(np.array([img for img, _ in dataset.imgs]))
            self.set_targets(np.array([target for _, target in dataset.imgs]))

    def get_num_classes(self) -> int:
        return 100
    
    def __getitem__(self, index) -> Tuple:
        data = self.get_data()[index]
        if not self.load_preprocessed:
            data = Image.open(data)
            data = data.convert("RGB")

        if self.transform is not None:
            data = self.transform(data)

        target = self.get_targets()[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

if __name__ == "__main__":
    from torch.utils.data import DataLoader


    root = "data"

    train_dataset = Imagenet100(
        root=root,
        split=Split.TRAIN,
        transform=tv.transforms.Compose(
            [
                tv.transforms.Resize((76, 76)),
                tv.transforms.CenterCrop((64, 64)),
                tv.transforms.ToTensor(),
            ]
        ),
        load_preprocessed=False
    )
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    imgs, labels = next(iter(train_loader))
    torch.save(imgs, os.path.join(root, "imagenet100", "train64x64_imgs"))
    torch.save(labels, os.path.join(root, "imagenet100", "train64x64_labels"))
    
    val_dataset = Imagenet100(
        root=root,
        split=Split.TEST,
        transform=tv.transforms.Compose(
            [
                tv.transforms.Resize((76, 76)),
                tv.transforms.CenterCrop((64, 64)),
                tv.transforms.ToTensor(),
            ]
        ),
        load_preprocessed=False
    )
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
    imgs, labels = next(iter(val_loader))
    torch.save(imgs, os.path.join(root, "imagenet100", "val64x64_imgs"))
    torch.save(labels, os.path.join(root, "imagenet100", "val64x64_labels"))
