from typing import Callable, Optional
from torchvision.datasets import SVHN

from .base import BaseDataset, Split


class SVHNDataset(BaseDataset):
    def __init__(
        self,
        root: str = "data",
        split: Split = Split.TRAIN,
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, split, download, transform, target_transform)
        if split == Split.TRAIN:
            split_str = "train"
        elif split == Split.TEST:
            split_str = "test"
        else:
            raise ValueError(f"{split} value not supported")
        dataset = SVHN(root=root, split=split_str, download=download)
        self.set_data(dataset.data)
        self.set_targets(dataset.labels)

    def get_num_classes(self) -> int:
        return 10
