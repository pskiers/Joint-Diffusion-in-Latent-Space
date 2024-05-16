from typing import Callable, Optional, Tuple
import numpy as np
import os
from PIL import Image
import torchvision as tv

from .base import BaseDataset, Split


class Imagenet100(BaseDataset):
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
        dataset = tv.datasets.ImageFolder(root=os.path.join(root, "imagenet100", "train" if split == Split.TRAIN else "val"))
        self.set_data(np.array([img for img, _ in dataset.imgs]))
        self.set_targets(np.array([target for _, target in dataset.imgs]))

    def get_num_classes(self) -> int:
        return 100
    
    def __getitem__(self, index) -> Tuple:
        path = self.get_data()[index]
        data = Image.open(path)
        img_array = np.array(data)
        if img_array.ndim == 2:
            img_array = img_array[np.newaxis, :, :]
        if img_array.shape[0] == 1:
            img_rgb_array = np.repeat(img_array, 3, axis=0)
            img_rgb_array = np.transpose(img_rgb_array, (1, 2, 0))
            data = Image.fromarray(img_rgb_array.astype('uint8'), 'RGB')

        if self.transform is not None:
            data = self.transform(data)

        target = self.get_targets()[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target
