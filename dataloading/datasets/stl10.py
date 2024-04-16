from typing import Callable, Optional, Tuple
from torchvision.datasets import STL10

from .base import BaseDataset, Split


class STL10Dataset(BaseDataset):
    def __init__(
        self,
        root: str = "data",
        split: Split = Split.TRAIN,
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__(root, split, download, transform, target_transform, **kwargs)
        if split == Split.TRAIN:
            split_str = "train"
        elif split == Split.TEST:
            split_str = "test"
        elif split == Split.UNLABELED:
            split_str = "unlabeled"
        else:
            raise ValueError(f"{split} value not supported")
        dataset = STL10(root=root, split=split_str, download=download)
        self.set_data(dataset.data)
        self.set_targets(dataset.labels)

    def __getitem__(self, index) -> Tuple:
        data = self.get_data()[index]
        data = data.transpose((1, 2, 0))
        if self.transform is not None:
            data = self.transform(data)

        # if self.split == Split.UNLABELED:
        #     return data

        target = self.get_targets()[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def get_num_classes(self) -> int:
        return 10
