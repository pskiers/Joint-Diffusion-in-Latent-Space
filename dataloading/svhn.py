import torch
import torchvision as tv


class AdjustedSVHN(torch.utils.data.Dataset):
    def __init__(self, train="train", labeled_per_class=None) -> None:
        super().__init__()
        self.dataset = tv.datasets.SVHN(root="./data", split=train, download=True, transform=tv.transforms.ToTensor())
        if labeled_per_class is not None and train == "train":
            labeled_samples = {i: 0 for i in range(10)}
            for i, label in enumerate(self.dataset.labels):
                labeled_samples[label] += 1
                if labeled_samples[label] > labeled_per_class:
                    self.dataset.labels[i] = -1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img.permute(1, 2, 0), label