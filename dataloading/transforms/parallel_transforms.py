from typing import List, Callable
import torch


class ParallelTransforms(object):
    def __init__(self, trnsfrms: List[Callable]): 
        self.trnsfrms = trnsfrms

    def __call__(self, img: torch.Tensor):
        return tuple(transform(img) for transform in self.trnsfrms)
