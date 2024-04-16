from typing import List
import torch


class Permute(object):
    def __init__(self, dims: List[int]):
        self.dims = dims

    def __call__(self, img: torch.Tensor):
        return img.permute(dims=self.dims)
