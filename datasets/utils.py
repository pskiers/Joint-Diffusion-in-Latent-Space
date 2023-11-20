from typing import Tuple, List
import numpy as np
import torch
import torch.utils.data as data


def equal_labels_random_split(
        dataset: data.Dataset,
        labels: List,
        amount_per_class: int,
        generator: torch.Generator = torch.default_generator
) -> Tuple[data.Subset, data.Subset]:
    indices_labels = []
    indices_rest = []
    per_class = {label: 0 for label in labels}

    permuted_indices = torch.randperm(
        len(dataset), generator=generator).tolist()
    for idx in permuted_indices:
        _, label = dataset[idx]
        per_class[label] += 1
        if per_class[label] <= amount_per_class:
            indices_labels.append(idx)
        else:
            indices_rest.append(idx)

    return (
        data.Subset(dataset, indices_rest),
        data.Subset(dataset, indices_labels)
    )


def cl_class_split(
        dataset: data.Dataset,
        tasks_labels: List[List]
) -> List[np.ndarray]:
    tasks_indices = [[] for _ in range(tasks_labels)]
    for i, (_, label) in enumerate(dataset):
        for j, task_labels in enumerate(tasks_labels):
            if label in task_labels:
                tasks_indices[j].append(i)
    return [
        np.array(task_indices)
        for task_indices in tasks_indices
    ]
