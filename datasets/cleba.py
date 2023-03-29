import torch
import torchvision as tv


class AdjustedCelbA(torch.utils.data.Dataset):
    def __init__(self, train="train") -> None:
        super().__init__()
        self.dataset = tv.datasets.CelebA(root="./data", train=train, download=True, transform=tv.transforms.ToTensor())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img.permute(1, 2, 0), label