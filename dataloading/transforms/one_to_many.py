from typing import Callable
import torch


class OneToMany(object):
    def __init__(self, transform: Callable, many: int):
        self.transform = transform
        self.many = many

    def __call__(self, img: torch.Tensor):
        out = [self.transform(img) for _ in range(self.many)]
        return torch.stack(out, dim=0)
