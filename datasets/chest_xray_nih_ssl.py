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
from .chest_xray_nih import ChestXRay_nih
from skmultilearn.model_selection import iterative_train_test_split
from .randaugment import RandAugmentMC

class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=256,
                                  padding=int(256*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=256,
                                  padding=int(256*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(x).squeeze(), self.normalize(weak).squeeze(), self.normalize(strong).squeeze()
    

class ChestXRay_nih_ssl(ChestXRay_nih):
    def __init__(self, mode='train', training_platform: str = 'local_sano',  img_size = 256, min_augmentation_ratio: int = 0.8, auto_augment = False, labeled = True) -> None:
        super().__init__(mode=mode, 
                         training_platform=training_platform,  
                         img_size = img_size, 
                         min_augmentation_ratio=min_augmentation_ratio, 
                         auto_augment = auto_augment)
        
        self.labeled = labeled
        X = self.final_image_df.loc[:, ["image_path"]]
        y = self.final_image_df.loc[:, [*self.labels, "no_finding"]]
        y = y.to_numpy()
        X = X.to_numpy()
        np.random.seed(402)
        self.X_unlabeled, self.y_unlabeled, self.X_labeled, self.y_labeled = iterative_train_test_split(X, y, test_size = 0.02)
        self.fixmatch_transform = TransformFixMatch(mean=0.5, std=0.5)
        
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
    ds = ChestXRay_nih_ssl(auto_augment=True, labeled=True, training_platform="de")
    print("labeled size", ds[1][0][1].shape, len(ds))

    ds = ChestXRay_nih_ssl(auto_augment=True, labeled=False, training_platform="de")
    print("unlabeled size", ds[1][0][1].shape, len(ds))