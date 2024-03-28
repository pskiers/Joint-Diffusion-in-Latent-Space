from .base import BaseDataset, Split, BaseTensorDataset
from .cifar10 import CIFAR10Dataset
from .cifar100 import CIFAR100Dataset
from .fashionMNIST import fashionMNISTDataset
from .mnist import MNISTDataset
from .stl10 import STL10Dataset
from .svhn import SVHNDataset


def str_to_split(string: str) -> Split:
    if string == "train":
        return Split.TRAIN
    elif string == "test":
        return Split.TEST
    elif string == "unlabeled":
        return Split.UNLABELED
    else:
        raise ValueError(f"{string} is not a valid split")


def get_dataset_cls(name: str):
    if name == "cifar10":
        return CIFAR10Dataset
    elif name == "cifar100":
        return CIFAR100Dataset 
    elif name == "fashionMNIST":
        return fashionMNISTDataset
    elif name == "MNIST":
        return MNISTDataset
    elif name == "stl10":
        return STL10Dataset
    elif name == "svhn":
        return SVHNDataset
    else:
        raise ValueError("Dataset not implemented")
