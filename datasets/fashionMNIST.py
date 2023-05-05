import torch
import torchvision as tv


class AdjustedFashionMNIST(torch.utils.data.Dataset):
    def __init__(self, train=False, labeled_per_class=None) -> None:
        super().__init__()
        self.dataset = tv.datasets.FashionMNIST(root="./data", train=train, download=True, transform=tv.transforms.Compose([tv.transforms.Resize(32) , tv.transforms.ToTensor()]))
        if labeled_per_class is not None and train is True:
            labeled_samples = {i: 0 for i in range(10)}
            for i, label in enumerate(self.dataset.targets):
                labeled_samples[int(label)] += 1
                if labeled_samples[int(label)] > labeled_per_class:
                    self.dataset.targets[i] = -1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return torch.concat([img, img, img], dim=0).permute(1, 2, 0), label