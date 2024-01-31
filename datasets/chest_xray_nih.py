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

class ChestXRay_nih(torch.utils.data.Dataset):
    def __init__(self, mode='train') -> None:
        super().__init__()
        data_path = "/home/jk/Joint-Diffusion-in-Latent-Space/chest_xray_nih"
        df = pd.read_csv(os.path.join(data_path, "Data_Entry_2017.csv"))[["Image Index", "Finding Labels"]]
        df = pd.concat([df.drop("Finding Labels", axis=1), df["Finding Labels"].str.get_dummies('|')], axis=1)
        df.rename(columns=lambda x: x.replace(" ", "_").lower(), inplace=True)
        self.labels = df.columns[~df.columns.isin(['no_finding', 'image_index'])]


        assert mode in ['val', 'train', 'test']

        if mode in ['val', 'train']:

            
            with open(os.path.join(data_path, "train_val_list.txt")) as file:
                image_list = [line.rstrip() for line in file]   
            df = df[df["image_index"].isin(image_list)]
            df["image_path"] = data_path+"/images256/"+df["image_index"]
            df.drop(columns=["image_index"], inplace=True)
            df = df.sample(frac=1, random_state=45654).reset_index(drop=True)
            
            self.split_idx = int(0.0*len(df))
            
            if mode =='val':
                self.final_image_df = df[:self.split_idx]

                self.transform = A.Compose([
                    #A.Resize(256,256),
                    A.Normalize(mean=0.5, std=0.5),
                ])
            elif mode =='train':
                self.final_image_df = df[self.split_idx:]

                # self.transform = A.Compose([
                #     A.RandomResizedCrop(width=256, height=256,scale=(0.9, 1.0), ratio=(0.9, 1.1)),
                #     A.HorizontalFlip(p=0.5),
                #     A.Normalize(mean=0.5, std=0.5),
                # ])

                self.transform = A.Compose([
                    A.RandomResizedCrop(width=256, height=256,scale=(0.3, 1.0), ratio=(0.75, 1.33)),
                    A.Rotate(15),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(mean=0.5, std=0.5),
                ])
            
        elif mode == 'test':
            with open(os.path.join(data_path, "test_list.txt")) as file:
                image_list = [line.rstrip() for line in file]   
            df = df[df["image_index"].isin(image_list)]
            df["image_path"] = data_path+"/images256/"+df["image_index"]
            df.drop(columns=["image_index"], inplace=True)
            self.final_image_df = df.sample(frac=1, random_state=45654).reset_index(drop=True)

            self.transform = A.Compose([
                    #A.Resize(256,256),
                    A.Normalize(mean=0.5, std=0.5),
                ])


        
    def __len__(self):
        return len(self.final_image_df)

    def __getitem__(self, index):

        img_path = self.final_image_df.loc[index, 'image_path']
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.uint8)
        image_transformed = self.transform(image=image)
        image = image_transformed["image"]
        image = torch.from_numpy(image)
        return image, self.final_image_df.loc[index, [*self.labels, "no_finding"]].to_numpy(dtype=int)
        
  

if __name__ == "__main__":
    ds = ChestXRay_nih()
    print(ds[1])