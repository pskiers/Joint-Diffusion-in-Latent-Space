import torch
import torchvision as tv


class GTSRB(torch.utils.data.Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.dataset = tv.datasets.ImageFolder(root=path, transform=tv.transforms.Compose([tv.transforms.Resize((32, 32)) , tv.transforms.ToTensor()]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img.permute(1, 2, 0), label