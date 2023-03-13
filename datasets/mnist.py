import torch
import torchvision as tv


class AdjustedMNIST(torch.utils.data.Dataset):
    def __init__(self, train=False) -> None:
        super().__init__()
        self.dataset = tv.datasets.MNIST(root="./data", train=train, download=True, transform=tv.transforms.Compose([tv.transforms.Resize(32) , tv.transforms.ToTensor()]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return torch.concat([img, img, img], dim=0).transpose(0, 2), label
