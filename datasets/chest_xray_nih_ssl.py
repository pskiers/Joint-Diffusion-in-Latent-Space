import torch
import torchvision as tv
import random
from glob import glob
import os
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets, models, transforms
import PIL
from chest_xray_nih import ChestXRay_nih
from skmultilearn.model_selection import iterative_train_test_split

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
        self.X_labeled, self.y_labeled, self.X_unlabeled, self.y_unlabeled = iterative_train_test_split(X, y, test_size = 0.02)
        
    def __len__(self):
        if self.labeled:
            return len(self.X_labeled)
        else:
            return len(self.X_unlabeled)
        

    def __getitem__(self, index):

        if self.labeled:
            img_path = self.X_labeled[index].item()
        else:
            img_path = self.X_unlabeled[index].item()

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

        if self.labeled:
            return image_transformed, self.y_labeled[index]
        else:
            return image_transformed, self.y_unlabeled[index]

if __name__ == "__main__":
    ds = ChestXRay_nih_ssl(auto_augment=True, labeled=False)
    print(ds[1][0].shape)