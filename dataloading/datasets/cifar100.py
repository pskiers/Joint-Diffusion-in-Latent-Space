from typing import Callable, Optional
from torchvision.datasets import CIFAR100
import numpy as np

from .base import BaseDataset, Split


class CIFAR100Dataset(BaseDataset):
    def __init__(
        self,
        root: str = "data",
        split: Split = Split.TRAIN,
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs
    ) -> None:
        super().__init__(root, split, download, transform, target_transform, **kwargs)
        if split == Split.TRAIN:
            train = True
        elif split == Split.TEST:
            train = False
        else:
            raise ValueError(f"{split} value not supported")
        dataset = CIFAR100(root=root, train=train, download=download)
        self.set_data(dataset.data)
        self.set_targets(np.array(dataset.targets))

    def get_num_classes(self) -> int:
        return 100
