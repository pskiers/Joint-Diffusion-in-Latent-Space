from typing import Tuple, Callable, Optional, Union
from enum import Enum
from abc import abstractmethod, ABC
import numpy as np
import torch
import torch.utils.data as data


class Split(Enum):
    TRAIN = 0
    TEST = 1
    UNLABELED = 2


class BaseDataset(data.Dataset, ABC):
    """
    Base dataset interface
    """

    @abstractmethod
    def __init__(
        self,
        root: str = "data",
        split: Split = Split.TRAIN,
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.download = download
        self.transform = transform
        self.target_transform = target_transform

        self.data: Union[torch.Tensor, np.ndarray]
        self.targets: Union[torch.Tensor, np.ndarray]

    def get_data(self) -> Union[torch.Tensor, np.ndarray]:
        return self.data

    def get_targets(self) -> Union[torch.Tensor, np.ndarray]:
        return self.targets

    def set_data(self, data: Union[torch.Tensor, np.ndarray]) -> None:
        self.data = data

    def set_targets(self, targets: Union[torch.Tensor, np.ndarray]) -> None:
        self.targets = targets

    def set_transforms(self, new_transforms: Callable) -> None:
        self.transform = new_transforms

    @abstractmethod
    def get_num_classes(self) -> int:
        """Returns number of classes in the dataset"""

    def __getitem__(self, index) -> Tuple:
        data = self.get_data()[index]
        target = self.get_targets()[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self) -> int:
        return len(self.get_data())


class BaseTensorDataset(BaseDataset):
    def __init__(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(transform=transform, target_transform=target_transform)
        self.data = data
        self.targets = targets
    
    def get_num_classes(self) -> int:
        return len(torch.unique(self.targets))
