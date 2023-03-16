import torch
import torchvision as tv


class AdjustedCIFAR10(torch.utils.data.Dataset):
    def __init__(self, train=False) -> None:
        super().__init__()
        self.dataset = tv.datasets.CIFAR10(root="./data", train=train, download=True, transform=tv.transforms.ToTensor())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img.transpose(0, 2), label