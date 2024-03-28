from typing import Optional, List, Tuple
import math
import numpy as np
from copy import deepcopy

from ..datasets import BaseDataset


def ssl_dataset_split(
    dataset: BaseDataset,
    num_labeled: int,
    min_labeled: Optional[int] = None,
    seed: int = 42,
) -> Tuple[BaseDataset, BaseDataset]:
    np.random.seed(seed)
    label_per_class = num_labeled // dataset.get_num_classes()
    labeled_idx: List[int] = []

    labels = np.array(dataset.targets)
    labeled_idx = []
    for i in range(dataset.get_num_classes()):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx_arr = np.array(labeled_idx)
    assert len(labeled_idx_arr) == num_labeled

    if min_labeled is not None and num_labeled < min_labeled:
        num_expand_x = math.ceil(min_labeled / num_labeled)
        labeled_idx_arr = np.hstack([labeled_idx_arr for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx_arr)
    labeled_idx = list(labeled_idx_arr)

    unlabeled_ds = dataset
    labeled_ds = deepcopy(dataset)
    new_data = labeled_ds.get_data()[labeled_idx]
    new_targets = labeled_ds.get_targets()[labeled_idx]
    labeled_ds.set_data(new_data)
    labeled_ds.set_targets(new_targets)

    return labeled_ds, unlabeled_ds
