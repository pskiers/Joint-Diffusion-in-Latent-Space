import torch
from torchvision import transforms
from typing import List, Callable
import kornia as K
import kornia.augmentation as aug

from .randaugment import RandAugmentMC


class JointDiffusionAugmentations(object):
    def __init__(
        self,
        img_size: int,
        mean: List[float] = [0.5, 0.5, 0.5],
        std: List[float] = [0.5, 0.5, 0.5],
    ) -> None:
        self.toTensor = transforms.ToTensor()
        self.augmentation = K.augmentation.ImageSequential(
            aug.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.25),
            aug.RandomResizedCrop((img_size, img_size), scale=(0.5, 1), p=0.25),
            aug.RandomRotation((-30, 30), p=0.25),
            aug.RandomHorizontalFlip(0.5),
            aug.RandomContrast((0.6, 1.8), p=0.25),
            aug.RandomSharpness((0.4, 2), p=0.25),
            aug.RandomBrightness((0.6, 1.8), p=0.25),
            aug.RandomMixUpV2(p=0.5),
            # random_apply=(1, 6),
            aug.Normalize(mean=mean, std=std),
        )

    def __call__(self, x):
        return self.augmentation(self.toTensor(x))


class TransformFixMatch(object):
    def __init__(
        self,
        mean: List[float] = [0.5, 0.5, 0.5],
        std: List[float] = [0.5, 0.5, 0.5],
    ):
        self.weak = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32, padding=int(32 * 0.125), padding_mode="reflect"
                ),
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32, padding=int(32 * 0.125), padding_mode="reflect"
                ),
                RandAugmentMC(n=2, m=10),
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(x), self.normalize(weak), self.normalize(strong)


class TransformRandAugmentSupervised(object):
    def __init__(
        self,
        mean: List[float] = [0.5, 0.5, 0.5],
        std: List[float] = [0.5, 0.5, 0.5],
    ):
        self.strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32, padding=int(32 * 0.125), padding_mode="reflect"
                ),
                RandAugmentMC(n=2, m=10),
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    def __call__(self, x):
        strong = self.strong(x)
        return self.normalize(x), self.normalize(strong)


class TransformMultiFixMatch(object):
    def __init__(self, mean, std, number=10):
        self.number = number
        self.weak = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode="reflect"),
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode="reflect"),
                RandAugmentMC(n=2, m=10),
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    def __call__(self, x):
        img = [self.normalize(x) for _ in range(self.number)]
        weak = [self.normalize(self.weak(x)) for _ in range(self.number)]
        strong = [self.normalize(self.strong(x)) for _ in range(self.number)]
        return (
            torch.stack(img, dim=0),
            torch.stack(weak, dim=0),
            torch.stack(strong, dim=0),
        )

predefined = [
    "joint_diffusion_augmentations",
    "flip_and_crop",
    "predefined_randaugment",
    "fixmatch_augmentations",
    "1_to_n_fixmatch_augmentations",
    "supervised_randaugment",
]

def get_predefined(name: str, kwargs: dict) -> Callable:
    if name == "joint_diffusion_augmentations":
        return JointDiffusionAugmentations(**kwargs)
    elif name == "flip_and_crop":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32, padding=int(32 * 0.125), padding_mode="reflect"
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=kwargs["mean"], std=kwargs["std"]),
            ]
        )
    elif name == "predefined_randaugment":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32, padding=int(32 * 0.125), padding_mode="reflect"
                ),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=kwargs["mean"], std=kwargs["std"]),
            ]
        )
    elif name == "fixmatch_augmentations":
        return TransformFixMatch(**kwargs)
    elif name == "1_to_n_fixmatch_augmentations":
        return TransformMultiFixMatch(**kwargs)
    elif name == "supervised_randaugment":
        return TransformRandAugmentSupervised(**kwargs)
    else:
        raise ValueError(f"No such predefined transform: {name}")
