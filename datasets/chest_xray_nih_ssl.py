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
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        # weak = self.weak(x)
        # strong = self.strong(x)
        return self.normalize(x).squeeze() #, self.normalize(weak).squeeze(), self.normalize(strong).squeeze()
    

class ChestXRay_nih_ssl(ChestXRay_nih):
    def __init__(self, mode='train', training_platform: str = 'local_sano', labeled = True) -> None:
        super().__init__(mode=mode, 
                         training_platform=training_platform,  
                        )
        print('SSL DATSDET LOADING')
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
        
        if self.labeled:
            if self.mode=="train":
                self.final_image_df = self.X_labeled_train.copy()
                self.final_label = self.y_labeled_train.copy()
            elif self.mode=="val":
                self.final_image_df = self.X_labeled_val.copy()
                self.final_label = self.y_labeled_val.copy()

        else:
            self.final_image_df = self.X_unlabeled.copy()[:5000]
            self.final_label = self.y_unlabeled.copy()[:5000]
        
        self.final_image_df = self.final_image_df
        self.final_label = self.final_label

        
        del self.X_unlabeled, self.y_unlabeled, X_labeled, self.X_labeled_train, self.y_labeled_train, self.X_labeled_val, self.y_labeled_val, X, y

        self.fixmatch_transform = TransformFixMatch(mean=0.5, std=0.5)
        
    def __len__(self):
        return len(self.final_image_df)
        

    def __getitem__(self, index):

        if self.labeled:
            img_path = self.final_image_df[index].item()
            label = self.final_label[index]
        
            image = PIL.Image.open(img_path)
            image = image.convert('L')
            image_transformed = self.transform(image).squeeze()
            return image_transformed, label

        else:
            img_path = self.final_image_df[index].item()
            image = PIL.Image.open(img_path)
            image = image.convert('L')
            image_transformed = self.fixmatch_transform(image)
            return image_transformed, self.final_label[index]

if __name__ == "__main__":
    ds = ChestXRay_nih_ssl(labeled=True, training_platform="local_sano")
    print("labeled size", ds[1][0][1].shape, len(ds))

    ds = ChestXRay_nih_ssl(labeled=False, training_platform="local_sano")
    print("unlabeled size", ds[1][0][1].shape, len(ds))

    ds = ChestXRay_nih_ssl(mode="val",labeled=True, training_platform="local_sano")
    print("val labeled size", ds[1][0][1].shape, len(ds))