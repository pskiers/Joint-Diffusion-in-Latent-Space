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
    def __init__(self, mode='train', training_platform: str = 'local_sano',  img_size = 256, pick_class = None) -> None:
        super().__init__()
        print('BBOX READ AS INT FIX IT')

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
        self.df = self.df.head(3)
        print(self.df)



        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=0.5, std=0.5),
        ])
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        img_path = self.df.loc[index, "Image Index"]
        img_path = os.path.join(self.data_path, "images256", img_path)
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.uint8)
        image_transformed = self.transform(image=image)
        image_transformed = image_transformed["image"]
        image_transformed = torch.from_numpy(image_transformed)

        return image_transformed, self.df.loc[index, ["x", "y", "w", "h"]].to_numpy(dtype=int), self.df.loc[index, "Finding Label"]
        
  

if __name__ == "__main__":
    ds = ChestXRay_nih_bbox(pick_class="Mass")
    print(ds.df)
    print(ds[1][0].shape, ds[1][1], ds[1][2])