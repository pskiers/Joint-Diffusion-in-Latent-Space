from typing import Tuple, Optional, List
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torchvision import datasets

from .cifar10 import AdjustedCIFAR10
from .cifar100 import AdjustedCIFAR100
from .mnist import AdjustedMNIST
from .cleba import AdjustedCelbA
from .fashionMNIST import AdjustedFashionMNIST
from .svhn import AdjustedSVHN
from .gtsrb import GTSRB
from .utils import equal_labels_random_split, cl_class_split
from .fixmatch_cifar import DATASET_GETTERS


@dataclass
class RandAugmentArgs:
    num_labeled: int = 1000
    num_classes: int = 10
    expand_labels: bool = True
    batch_size: int = 64
    eval_step: int = 1024


def get_dataloaders(
    name: str,
    train_batches: Tuple[int],
    val_batch: int,
    num_workers: int,
    num_labeled: Optional[int] = None,
):
    if name == "cifar10":
        train_ds = AdjustedCIFAR10(train=True)
        val_ds = AdjustedCIFAR10(train=False)
        num_classes = 10
        return non_randaugment_dl(
            train_ds, val_ds, num_labeled, train_batches, val_batch, num_classes, num_workers
        )
    elif name == "cifar10_randaugment":
        if num_labeled is not None:
            if len(train_batches) != 1:
                raise ValueError(
                    "Need 1 train batch size - supervised batch size; unsupervised bs = train bs * 7"
                )
            args = RandAugmentArgs(
                num_labeled=num_labeled, num_classes=10, batch_size=train_batches[0]
            )
            labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS["cifar10"](
                args, "./data"
            )
            return ssl_randaugment_dl(
                labeled_dataset,
                unlabeled_dataset,
                test_dataset,
                train_batches[0],
                val_batch,
                num_workers,
            )
        else:
            if len(train_batches) != 1:
                raise ValueError("Need 1 train batch size - supervised batch size")
            args = RandAugmentArgs(
                num_labeled=num_labeled, num_classes=10, batch_size=train_batches[0]
            )
            labeled_dataset, test_dataset = DATASET_GETTERS["cifar10_supervised"](args, "./data")
            return randaugment_dl(
                labeled_dataset, test_dataset, train_batches[0], val_batch, num_workers
            )
    elif name == "cifar100":
        train_ds = AdjustedCIFAR100(train=True)
        val_ds = AdjustedCIFAR100(train=False)
        num_classes = 100
        return non_randaugment_dl(
            train_ds, val_ds, num_labeled, train_batches, val_batch, num_classes, num_workers
        )
    elif name == "cifar100_randaugment":
        if num_labeled is not None:
            if len(train_batches) != 1:
                raise ValueError(
                    "Need 1 train batch size - supervised batch size; unsupervised bs = train bs * 7"
                )
            args = RandAugmentArgs(
                num_labeled=num_labeled, num_classes=100, batch_size=train_batches[0]
            )
            labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS["cifar100"](
                args, "./data"
            )
            return ssl_randaugment_dl(
                labeled_dataset,
                unlabeled_dataset,
                test_dataset,
                train_batches[0],
                val_batch,
                num_workers,
            )
        else:
            if len(train_batches) != 1:
                raise ValueError("Need 1 train batch size - supervised batch size")
            args = RandAugmentArgs(
                num_labeled=num_labeled, num_classes=10, batch_size=train_batches[0]
            )
            labeled_dataset, test_dataset = DATASET_GETTERS["cifar100_supervised"](args, "./data")
            return randaugment_dl(
                labeled_dataset, test_dataset, train_batches[0], val_batch, num_workers
            )
    elif name == "svhn":
        train_ds = AdjustedSVHN(train="train")
        val_ds = AdjustedSVHN(train="test")
        num_classes = 10
        return non_randaugment_dl(
            train_ds, val_ds, num_labeled, train_batches, val_batch, num_classes, num_workers
        )
    elif name == "svhn_randaugment":
        if num_labeled is not None:
            if len(train_batches) != 1:
                raise ValueError(
                    "Need 1 train batch size - supervised batch size; unsupervised bs = train bs * 7"
                )
            args = RandAugmentArgs(
                num_labeled=num_labeled, num_classes=10, batch_size=train_batches[0]
            )
            labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS["svhn"](
                args, "./data"
            )
            return ssl_randaugment_dl(
                labeled_dataset,
                unlabeled_dataset,
                test_dataset,
                train_batches[0],
                val_batch,
                num_workers,
            )
        else:
            if len(train_batches) != 1:
                raise ValueError("Need 1 train batch size - supervised batch size")
            args = RandAugmentArgs(
                num_labeled=num_labeled, num_classes=10, batch_size=train_batches[0]
            )
            labeled_dataset, test_dataset = DATASET_GETTERS["svhn_supervised"](args, "./data")
            return randaugment_dl(
                labeled_dataset, test_dataset, train_batches[0], val_batch, num_workers
            )
    elif name == "mnist":
        train_ds = AdjustedMNIST(train=True)
        val_ds = AdjustedMNIST(train=False)
        num_classes = 10
        return non_randaugment_dl(
            train_ds, val_ds, num_labeled, train_batches, val_batch, num_classes, num_workers
        )
    elif name == "fashion_mnist":
        train_ds = AdjustedFashionMNIST(train=True)
        val_ds = AdjustedFashionMNIST(train=False)
        num_labeled = 10
        return non_randaugment_dl(
            train_ds, val_ds, num_labeled, train_batches, val_batch, num_classes, num_workers
        )
    elif name == "stl10":
        if len(train_batches) != 1:
            raise ValueError(
                "Need 1 train batch size - supervised batch size; unsupervised bs = train bs * 7"
            )
        args = RandAugmentArgs(num_labeled=num_labeled, num_classes=10, batch_size=train_batches[0])
        labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS["cifar10"](
            args, "./data"
        )
        return ssl_randaugment_dl(
            labeled_dataset,
            unlabeled_dataset,
            test_dataset,
            train_batches[0],
            val_batch,
            num_workers,
        )
    else:
        raise NotImplementedError(f"Dataset {name} not implemented")


def supervised_randaugment_dl(labeled_dataset, test_dataset, batch_train, batch_val, num_workers):
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=RandomSampler(labeled_dataset),
        batch_size=batch_train,
        num_workers=num_workers,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_val,
        num_workers=num_workers,
    )
    return labeled_trainloader, test_loader


def ssl_randaugment_dl(
    labeled_dataset, unlabeled_dataset, test_dataset, batch_train, batch_val, num_workers
):
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=RandomSampler(labeled_dataset),
        batch_size=batch_train,
        num_workers=num_workers,
        drop_last=True,
    )

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=RandomSampler(unlabeled_dataset),
        batch_size=batch_train * 7,
        num_workers=num_workers,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_val,
        num_workers=num_workers,
    )
    return (labeled_trainloader, unlabeled_trainloader), test_loader


def randaugment_dl(labeled_dataset, test_dataset, batch_train, batch_val, num_workers):
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=RandomSampler(labeled_dataset),
        batch_size=batch_train,
        num_workers=num_workers,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_val,
        num_workers=num_workers,
    )
    return labeled_trainloader, test_loader


def non_randaugment_dl(
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    num_labeled: int,
    train_batches: Tuple[int],
    val_batch: int,
    num_classes: int,
    num_workers: int,
):
    if num_labeled is not None:
        if len(train_batches) != 2:
            raise ValueError("Need 2 train batch sizes - unsupervised and supervised batch size")
        _, train_supervised = equal_labels_random_split(
            train_ds,
            labels=[i for i in range(num_classes)],
            amount_per_class=num_labeled // num_classes,
            generator=torch.Generator().manual_seed(42),
        )
        train_dl_unsupervised = torch.utils.data.DataLoader(
            train_ds,
            batch_size=train_batches[0],
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )
        train_dl_supervised = torch.utils.data.DataLoader(
            train_supervised,
            batch_size=train_batches[1],
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )
        train_dl = (train_dl_supervised, train_dl_unsupervised)
    else:
        if len(train_batches) != 1:
            raise ValueError("Need 1 train batch size - supervised batch size")
        train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=train_batches[0],
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

    valid_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=val_batch, shuffle=False, num_workers=num_workers
    )
    return train_dl, valid_dl


def get_cl_datasets(
    name: str,
    num_labeled: int,
    sup_batch: int,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
):
    if name == "cifar10_randaugment":
        if num_labeled is not None:
            args = RandAugmentArgs(num_labeled=num_labeled, num_classes=10, batch_size=sup_batch)
            labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS["cifar10"](
                args, "./data", labels_to_tensor=True, mean=mean, std=std
            )
            tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
            sup_tasks_indices = cl_class_split(labeled_dataset.targets, tasks)
            unsup_tasks_indices = cl_class_split(unlabeled_dataset.targets, tasks)
            tasks_datasets = [
                (Subset(labeled_dataset, sup_task_idx), Subset(unlabeled_dataset, unsup_task_idx))
                for sup_task_idx, unsup_task_idx in zip(sup_tasks_indices, unsup_tasks_indices)
            ]
            return tasks_datasets, test_dataset, tasks
        else:
            raise NotImplementedError
    elif name == "cifar100_randaugment":
        if num_labeled is not None:
            args = RandAugmentArgs(num_labeled=num_labeled, num_classes=100, batch_size=sup_batch)
            labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS["cifar100"](
                args, "./data", labels_to_tensor=True
            )
            tasks = [[j for j in range(i * 10, (i + 1) * 10)] for i in range(10)]
            sup_tasks_indices = cl_class_split(labeled_dataset.targets, tasks)
            unsup_tasks_indices = cl_class_split(unlabeled_dataset.targets, tasks)
            tasks_datasets = [
                (Subset(labeled_dataset, sup_task_idx), Subset(unlabeled_dataset, unsup_task_idx))
                for sup_task_idx, unsup_task_idx in zip(sup_tasks_indices, unsup_tasks_indices)
            ]
            return tasks_datasets, test_dataset, tasks
        else:
            raise NotImplementedError
    elif name == "cifar10":
        train_ds = AdjustedCIFAR10(train=True)
        val_ds = AdjustedCIFAR10(train=False)
        tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        train_indices = cl_class_split(train_ds.dataset.targets, tasks)
        tasks_datasets = [Subset(train_ds, task_idx) for task_idx in train_indices]
        return tasks_datasets, val_ds, tasks
    else:
        raise NotImplementedError(f"Dataset {name} not implemented")