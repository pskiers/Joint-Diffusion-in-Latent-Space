from typing import Dict, List, Tuple, Callable, Union
from torch.utils.data import DataLoader, RandomSampler
from .splitting import ssl_dataset_split, cl_dataset_split
from .datasets import (
    get_dataset_cls,
    str_to_split,
    Split,
    BaseDataset,
    BaseTensorDataset,
)
from .transforms import get_transform


def make_dataloader(
    dataset: BaseDataset,
    batch_size: int,
    num_workers: int = 0,
    sampler: str = "none",
    shuffle: bool = True,
) -> DataLoader:
    sampl = (
        None
        if sampler == "none"
        else RandomSampler(dataset, num_samples=max(len(dataset), batch_size * 500))
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampl,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def _get_datasets(name: str, kwargs: Dict) -> Tuple[List[BaseDataset], List[int]]:
    ds = []
    bs = []
    if name == "ssl_split":
        sup_kwargs = kwargs["supervised"]
        unsup_kwargs = kwargs["unsupervised"]
        labeled_ds, unlabeled_ds = get_ssl_datasets(kwargs)
        ds.append(labeled_ds)
        ds.append(unlabeled_ds)
        bs.append(sup_kwargs["batch_size"])
        bs.append(unsup_kwargs["batch_size"])
    elif name == "dataset":
        d = _get_dataset(kwargs)
        ds.append(d)
        bs.append(kwargs["batch_size"])
    return ds, bs


def get_datasets(config: Dict) -> Tuple:
    train_ds: List = []
    train_bs: List = []
    train = config["train"]
    tasks = []
    for item in train:
        key, kwargs = next(iter(item.items()))
        if key == "cl_split":
            cl_ds = []
            tasks = kwargs["tasks"]
            for d in kwargs["datasets"]:
                key, ds_kwargs = next(iter(d.items()))
                ds, bs = _get_datasets(key, ds_kwargs)
                cl_ds.extend(ds)
                train_bs.extend(bs)
            split_ds = cl_dataset_split(cl_ds, kwargs["tasks"])
            train_ds.extend(split_ds)
        else:
            ds, bs = _get_datasets(key, kwargs)
            train_ds.extend(ds)
            train_bs.extend(bs)

    val = config["validation"]
    val_ds = get_dataset_cls(val.get("name"))(
        root=val.pop("root", "data"),
        split=str_to_split(val.pop("split", Split.TEST)),
        download=val.pop("download", True),
        **kwargs
    )
    val_transforms = get_transform(val["transforms"])
    val_ds.set_transforms(val_transforms)
    val_bs = val["batch_size"]

    if len(train_ds) == 1:
        return train_ds[0], train_bs[0], val_ds, val_bs, tasks
    return train_ds, train_bs, val_ds, val_bs, tasks


def get_dataloaders(
    config: Dict,
) -> Tuple[Union[List[DataLoader], DataLoader], DataLoader]:
    train_dataloaders: List[DataLoader] = []

    train = config["train"]
    for item in train:
        key, kwargs = next(iter(item.items()))
        if key == "ssl_split":
            sup_kwargs = kwargs["supervised"]
            unsup_kwargs = kwargs["unsupervised"]
            labeled_ds, unlabeled_ds = get_ssl_datasets(kwargs)

            labeled_dl = make_dataloader(
                dataset=labeled_ds,
                batch_size=sup_kwargs["batch_size"],
                num_workers=sup_kwargs.get("num_workers", 0),
                sampler=sup_kwargs.get("sampler", "none"),
                shuffle=sup_kwargs.get("shuffle", True),
            )

            unlabeled_dl = make_dataloader(
                dataset=unlabeled_ds,
                batch_size=unsup_kwargs["batch_size"],
                num_workers=unsup_kwargs.get("num_workers", 0),
                sampler=unsup_kwargs.get("sampler", "none"),
                shuffle=unsup_kwargs.get("shuffle", True),
            )

            train_dataloaders.append(labeled_dl)
            train_dataloaders.append(unlabeled_dl)
        elif key == "dataset":
            ds = _get_dataset(kwargs)
            dataloader = make_dataloader(
                dataset=ds,
                batch_size=kwargs["batch_size"],
                num_workers=kwargs.get("num_workers", 0),
                sampler=kwargs.get("sampler", "none"),
                shuffle=kwargs.get("shuffle", True),
            )
            train_dataloaders.append(dataloader)

    val = config["validation"]
    val_ds = get_dataset_cls(val.get("name"))(
        root=val.pop("root", "data"),
        split=str_to_split(val.pop("split", Split.TEST)),
        download=val.pop("download", True),
        **kwargs
    )
    val_transforms = get_transform(val["transforms"])
    val_ds.set_transforms(val_transforms)
    val_dataloader = make_dataloader(
        dataset=val_ds,
        batch_size=val["batch_size"],
        num_workers=val.get("num_workers", 0),
        sampler=val.get("sampler", "none"),
        shuffle=val.get("shuffle", True),
    )

    if len(train_dataloaders) == 1:
        return train_dataloaders[0], val_dataloader
    return train_dataloaders, val_dataloader


def _get_dataset(kwargs):
    ds_transforms = get_transform(kwargs.pop("transforms"))
    ds = get_dataset_cls(kwargs.get("name"))(
        root=kwargs.pop("root", "data"),
        split=str_to_split(kwargs.pop("split", Split.TRAIN)),
        download=kwargs.pop("download", True),
        **kwargs
    )
    ds.set_transforms(ds_transforms)
    return ds


def get_ssl_datasets(kwargs):
    sup_kwargs = kwargs["supervised"]
    unsup_kwargs = kwargs["unsupervised"]
    dataset = get_dataset_cls(kwargs.get("name"))(
        root=kwargs.pop("root", "data"),
        split=str_to_split(kwargs.pop("split", Split.TRAIN)),
        download=kwargs.pop("download", True),
        **kwargs
    )
    labeled_ds, unlabeled_ds = ssl_dataset_split(
        dataset=dataset,
        num_labeled=kwargs["num_labeled"],
        min_labeled=kwargs.get("min_labeled", None),
        equal_labels=kwargs.get("equal_labels", True),
        seed=kwargs.get("seed", 42),
    )
    labeled_transforms = get_transform(sup_kwargs["transforms"])
    labeled_ds.set_transforms(labeled_transforms)

    unlabeled_transforms = get_transform(unsup_kwargs["transforms"])
    unlabeled_ds.set_transforms(unlabeled_transforms)
    return labeled_ds, unlabeled_ds
