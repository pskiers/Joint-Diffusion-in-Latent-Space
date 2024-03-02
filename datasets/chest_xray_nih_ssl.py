import torch
import torchvision as tv
import random
from glob import glob
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets, models, transforms
import PIL
from PIL import ImageFilter
from .chest_xray_nih import ChestXRay_nih
from skmultilearn.model_selection import iterative_train_test_split
from .randaugment import RandAugmentMC


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TransformFixMatch(object):
    def __init__(self, mean, std, strong_transform=None, weak_transforms=None):
        if weak_transforms is None:
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=256,
                                    padding=int(256*0.125),
                                    padding_mode='reflect')])
        else:
            self.weak = weak_transforms
        if strong_transform is None:
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=256,
                                    padding=int(256*0.125),
                                    padding_mode='reflect'),
                RandAugmentMC(n=2, m=10)])
        else:
            self.strong = strong_transform
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(x).squeeze(), self.normalize(weak).squeeze(), self.normalize(strong).squeeze()


class ChestXRay_nih_ssl(ChestXRay_nih):
    def __init__(self, mode='train', training_platform: str = 'local_sano',  img_size=256, min_augmentation_ratio: int = 0.8, augment_type="fixmatch", labeled=True) -> None:
        super().__init__(mode=mode,
                         training_platform=training_platform,
                         img_size=img_size,
                         min_augmentation_ratio=min_augmentation_ratio,
                         auto_augment=False)

        self.labeled = labeled
        X = self.final_image_df.loc[:, ["image_path"]]
        y = self.final_image_df.loc[:, [*self.labels, "no_finding"]]
        y = y.to_numpy()
        X = X.to_numpy()
        np.random.seed(402)
        self.X_unlabeled, self.y_unlabeled, self.X_labeled, self.y_labeled = iterative_train_test_split(X, y, test_size=0.02)
        assert augment_type in ["fixmatch", "med_augmentations"]
        if augment_type == "fixmatch":
            self.fixmatch_transform = TransformFixMatch(mean=0.5, std=0.5)
        elif augment_type == "med_augmentations":
            strong_aug = transforms.Compose(
                [
                    transforms.RandomResizedCrop(256, scale=(0.2, 1.0)),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.8
                    ),
                    transforms.RandomRotation(45),
                    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                    # transforms.RandomRotation(5),
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomErasing(inplace=True)
                ]
            )
            weak_aug = transforms.Compose(
                [
                    transforms.RandomResizedCrop(256, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(),
                ]
            )
            self.fixmatch_transform = TransformFixMatch(mean=0.5, std=0.5, strong_transform=strong_aug, weak_transforms=weak_aug)

    def __len__(self):
        if self.labeled:
            return len(self.X_labeled)
        else:
            return len(self.X_unlabeled)

    def __getitem__(self, index):

        if self.labeled:
            img_path = self.X_labeled[index].item()
            if self.auto_augment:
                image = PIL.Image.open(img_path)
                image = image.convert('L')
                image_transformed = self.transform(image).squeeze()

            else:
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image = image.astype(np.uint8)
                image_transformed = self.transform(image=image)
                image_transformed = image_transformed["image"]
                image_transformed = torch.from_numpy(image_transformed)
            return image_transformed, self.y_labeled[index]

        else:
            img_path = self.X_unlabeled[index].item()
            image = PIL.Image.open(img_path)
            image = image.convert('L')
            image_transformed = self.fixmatch_transform(image)
            return image_transformed, self.y_unlabeled[index]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ds = ChestXRay_nih_ssl(auto_augment=True, labeled=True, training_platform="local_sano")
    print("labeled size", ds[1][0][1].shape, len(ds))
    # plt.imshow(ds[1][0][1].permute(1, 2, 0))
    # plt.show()

    ds = ChestXRay_nih_ssl(auto_augment=True, labeled=False, training_platform="local_sano")
    print("unlabeled size", ds[1][0][1].shape, len(ds))
    # plt.imshow(ds[1][0][1].permute(1, 2, 0))
    # plt.show()
