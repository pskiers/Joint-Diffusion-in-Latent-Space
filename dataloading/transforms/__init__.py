from typing import Callable, Union, Dict, List
from torchvision import transforms

from .one_to_many import OneToMany
from .parallel_transforms import ParallelTransforms
from .permute import Permute
from .predefined import get_predefined, predefined
from .randaugment import RandAugmentMC
from .torchvision_transforms import get_torchvision_transform


def get_transform(kwargs: Union[Dict, List]) -> Callable:
    if isinstance(kwargs, list):
        return transforms.Compose(
            [parse_transform(*next(iter(trans.items()))) for trans in kwargs]
        )
    else:
        return transforms.Compose([parse_transform(n, k) for n, k in kwargs.items()])


def parse_transform(name: str, kwargs: Union[Dict, List]) -> Callable:
    if name == "OneToMany":
        assert isinstance(kwargs, dict) or kwargs is None
        kwargs = kwargs if kwargs is not None else {}
        trans = transforms.Compose(
            [parse_transform(*next(iter(t.items()))) for t in kwargs["transforms"]]
        )
        return OneToMany(transform=trans, many=kwargs["many"])
    elif name == "ParallelTransforms":
        assert isinstance(kwargs, list)
        trans = [
            transforms.Compose(
                [parse_transform(*next(iter(t.items()))) for t in trnsfrm]
            )
            for trnsfrm in kwargs
        ]
        return ParallelTransforms(trans)
    elif name == "Permute":
        assert isinstance(kwargs, dict) or kwargs is None
        kwargs = kwargs if kwargs is not None else {}
        return Permute(**kwargs)
    elif name == "RandAugment":
        assert isinstance(kwargs, dict) or kwargs is None
        kwargs = kwargs if kwargs is not None else {}
        return RandAugmentMC(**kwargs)
    elif name in predefined:
        assert isinstance(kwargs, dict) or kwargs is None
        kwargs = kwargs if kwargs is not None else {}
        return get_predefined(name, kwargs)
    else:
        assert isinstance(kwargs, dict) or kwargs is None
        kwargs = kwargs if kwargs is not None else {}
        return get_torchvision_transform(name, kwargs)
