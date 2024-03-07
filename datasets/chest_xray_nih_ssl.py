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
from chest_xray_nih import ChestXRay_nih
from skmultilearn.model_selection import iterative_train_test_split
from randaugment import RandAugmentMC

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
    def __init__(self, mode='train', training_platform: str = 'local_sano', labeled = True) -> None:
        super().__init__(mode=mode, 
                         training_platform=training_platform,  
                        )
        
        self.mode=mode
        self.labeled = labeled
        if not self.labeled and self.mode=="val":
            raise "Unlabeled SSL val loader doesnt make sense."
        
        X = self.train_val_image_df.loc[:, ["image_path"]]
        y = self.train_val_image_df.loc[:, [*self.labels, "no_finding"]]
        y = y.to_numpy()
        X = X.to_numpy()
        np.random.seed(402)
        self.X_unlabeled, self.y_unlabeled, X_labeled, y_labeled = iterative_train_test_split(X, y, test_size = 0.02)
        self.X_labeled_train, self.y_labeled_train, self.X_labeled_val, self.y_labeled_val = iterative_train_test_split(X_labeled, y_labeled, test_size = 0.33)
        self.fixmatch_transform = TransformFixMatch(mean=0.5, std=0.5)
        
    def __len__(self):
        if self.labeled:
            if self.mode=="train":
                return len(self.X_labeled_train)
            elif self.mode=="val":
                return len(self.X_labeled_val)
        else:
            return len(self.X_unlabeled)
        

    def __getitem__(self, index):

        if self.labeled:
            if self.mode=="train":
                img_path = self.X_labeled_train[index].item()
                label = self.y_labeled_train[index]
            elif self.mode=="val":
                img_path = self.X_labeled_val[index].item()
                label = self.y_labeled_val[index]
            
            image = PIL.Image.open(img_path)
            image = image.convert('L')
            image_transformed = self.transform(image).squeeze()
            return image_transformed, label

        else:
            img_path = self.X_unlabeled[index].item()
            image = PIL.Image.open(img_path)
            image = image.convert('L')
            image_transformed = self.fixmatch_transform(image)
            return image_transformed, self.y_unlabeled[index]

if __name__ == "__main__":
    ds = ChestXRay_nih_ssl(labeled=True, training_platform="local_sano")
    print("labeled size", ds[1][0][1].shape, len(ds))

    ds = ChestXRay_nih_ssl(labeled=False, training_platform="local_sano")
    print("unlabeled size", ds[1][0][1].shape, len(ds))

    ds = ChestXRay_nih_ssl(mode="val",labeled=True, training_platform="local_sano")
    print("val labeled size", ds[1][0][1].shape, len(ds))