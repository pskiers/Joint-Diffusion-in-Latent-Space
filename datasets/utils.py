import torch


def equal_labels_random_split(dataset, labels, amount_per_class, generator=torch.default_generator):
    indices_labels = []
    indices_rest = []
    per_class = {label: 0 for label in labels}

    permuted_indices = torch.randperm(len(dataset), generator=generator).tolist()
    for idx in permuted_indices:
        _, label = dataset[idx]
        per_class[label] += 1
        if per_class[label] <= amount_per_class:
            indices_labels.append(idx)
        else:
            indices_rest.append(idx)

    return torch.utils.data.Subset(dataset, indices_rest), torch.utils.data.Subset(dataset, indices_labels)

