import torch
import torchvision as tv


class AdjustedCIFAR100(torch.utils.data.Dataset):
    def __init__(self, train=False) -> None:
        super().__init__()
        self.dataset = tv.datasets.CIFAR100(root="./data", train=train, download=True, transform=tv.transforms.ToTensor())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img.permute(1, 2, 0), label
