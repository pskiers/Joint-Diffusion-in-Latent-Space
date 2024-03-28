from typing import List, Union, Iterable
from copy import deepcopy

from ..datasets import BaseDataset


def get_cl_indices(labels: Iterable, task_labels: List[int]) -> List[int]:
    tasks_indices = []
    for i, label in enumerate(labels):
        if label in task_labels:
            tasks_indices.append(i)
    return tasks_indices


def cl_dataset_split(
    datasets: Union[List[BaseDataset]], tasks: List[List[int]]
) -> List[Union[List[BaseDataset], BaseDataset]]:
    all_datasets: List[Union[List[BaseDataset], BaseDataset]] = []
    for task in tasks:
        task_datasets: List[BaseDataset] = []
        for dataset in datasets:
            task_ds = deepcopy(dataset)
            indices = get_cl_indices(task_ds.get_targets(), task)
            task_ds.set_data(task_ds.get_data()[indices])
            task_ds.set_targets(task_ds.get_targets()[indices])
            task_datasets.append(task_ds)
        if len(task_datasets) == 0:
            all_datasets.append(task_datasets[0])
        else:
            all_datasets.append(task_datasets)
    return all_datasets
