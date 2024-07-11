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

class ChestXRay_nih_bbox(torch.utils.data.Dataset):
    def __init__(self, mode='train', training_platform: str = 'local_sano', pick_class = None, exclude_class = None) -> None:
        super().__init__()

        assert training_platform in ['plgrid', 'local_sano',]
        if training_platform=='plgrid':
            data_path = f"{os.environ['SCRATCH']}/chest_xray_nih"
        elif training_platform=='local_sano':
            data_path = "/home/jk/Joint-Diffusion-in-Latent-Space/chest_xray_nih"
        
        self.data_path = "/home/jk/Joint-Diffusion-in-Latent-Space/chest_xray_nih"
        self.df = pd.read_csv(os.path.join(data_path, "BBox_List_2017.csv")).iloc[:, :6]
        self.df.columns = ["Image Index", "Finding Label", "x", "y", "w", "h"]
        if pick_class:
            self.df = self.df[self.df["Finding Label"]==pick_class].reset_index(drop=True)
        if exclude_class:
            # Find the image names that contain the exclude class
            exclude_image_names = self.df[self.df["Finding Label"] == exclude_class]["Image Index"].unique()
            self.df = self.df[~self.df["Image Index"].isin(exclude_image_names)].reset_index(drop=True)

            # we cant do that - some images have several classes but they are in separate rows
            #self.df = self.df[self.df["Finding Label"]!=exclude_class].reset_index(drop=True)

        transformList = []
        transformList.append(transforms.Resize(256))
        transformList.append(transforms.ToTensor())
        transformList.append(transforms.Normalize(mean=0.5, std=0.5))  
        self.transform=transforms.Compose(transformList)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        img_path = self.df.loc[index, "Image Index"]
        img_path = os.path.join(self.data_path, "images", img_path)
        
        image = PIL.Image.open(img_path)
        image = image.convert('L')
        image_transformed = self.transform(image).squeeze()

        return image_transformed, self.df.loc[index, ["x", "y", "w", "h"]].to_numpy(dtype=int), self.df.loc[index, "Finding Label"]
        
  

if __name__ == "__main__":
    ds = ChestXRay_nih_bbox(pick_class="Mass")
    print(ds.df)
    print(ds[1][0].shape, ds[1][1], ds[1][2])