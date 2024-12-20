from typing import Callable, Optional, Tuple
import torch

from .base import BaseDataset, Split


class NormalDistribution(BaseDataset):
    def __init__(
        self,
        root: str = "data",
        split: Split = Split.TRAIN,
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        dim: int = 2,
        mean: float | tuple[float] = 0.0,
        std: float | tuple[float] = 1.0,
        **kwargs
    ) -> None:
        super().__init__(root, split, download, transform, target_transform, **kwargs)
        assert (isinstance(mean, float) and dim == 1) or (len(mean) == dim)
        assert (isinstance(std, float) and dim == 1) or (len(std) == dim)
        self.dim = dim
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        
        data = (torch.rand(1000000, self.dim) * 2 - 1).reshape((1000000, self.dim, 1, 1))
        self.set_data(data)
        self.set_targets(torch.zeros((len(data),)))
    
    # def __getitem__(self, index) -> Tuple:
    #     # data = torch.normal(mean=self.mean, std=self.std).reshape((-1, 1, 1))
    #     data = (torch.rand(self.dim) * 2 - 1).reshape((-1, 1, 1))
    #     target = 0

    #     if self.transform is not None:
    #         data = self.transform(data)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     return data, target

    # def __len__(self) -> int:
    #     return 50000

    def get_num_classes(self) -> int:
        return 1


class SomePointsDistribution(BaseDataset):
    def __init__(
        self,
        root: str = "data",
        split: Split = Split.TRAIN,
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        points: list[tuple[float]] = [(0.0,)], 
        **kwargs
    ) -> None:
        super().__init__(root, split, download, transform, target_transform, **kwargs)
        data = torch.tensor(points).reshape((len(points), -1, 1, 1))
        if len(data) < 10000:
            data = torch.cat([data] * 10000)
        self.set_data(data)
        self.set_targets(torch.zeros((len(data),)))

    def get_num_classes(self):
        return 1
