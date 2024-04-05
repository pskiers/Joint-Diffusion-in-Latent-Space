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

class ChestXRay_nih(torch.utils.data.Dataset):
    def __init__(self, mode='train', training_platform: str = 'local_sano') -> None:
        super().__init__()

        assert training_platform in ['plgrid', 'local_sano', "de"]
        if training_platform=='plgrid':
            self.data_path = f"{os.environ['SCRATCH']}/chest_xray_nih"
        elif training_platform=='local_sano':
            self.data_path = "/home/jk/Joint-Diffusion-in-Latent-Space/chest_xray_nih"
        elif training_platform=='de':
            self.data_path = "/data/jan_dubinski/Joint-Diffusion-in-Latent-Space/sano_machines/data"
        
        df = pd.read_csv(os.path.join(self.data_path, "Data_Entry_2017.csv"))[["Image Index", "Finding Labels"]]
        df = pd.concat([df.drop("Finding Labels", axis=1), df["Finding Labels"].str.get_dummies('|')], axis=1)
        df.rename(columns=lambda x: x.replace(" ", "_").lower(), inplace=True)
        self.labels = df.columns[~df.columns.isin(['no_finding', 'image_index'])]

        assert mode in ['val', 'train', 'test']

        if mode in ['val', 'train']:

            
            with open(os.path.join(self.data_path, "train_val_list.txt")) as file:
                image_list = [line.rstrip() for line in file]   
            df = df[df["image_index"].isin(image_list)]
            df["image_path"] = self.data_path+"/images/"+df["image_index"]
            df.drop(columns=["image_index"], inplace=True)
            df = df.sample(frac=1, random_state=45654).reset_index(drop=True)
            self.split_idx = int(0.1*len(df))
            self.train_val_image_df = df
            
            if mode =='val':
                self.final_image_df = df[:self.split_idx].reset_index(drop=True)

                transformList = []
                transformList.append(transforms.Resize(256))
                transformList.append(transforms.ToTensor())
                transformList.append(transforms.Normalize(mean=0.5, std=0.5))  
                self.transform=transforms.Compose(transformList)

            elif mode =='train':
                self.final_image_df = df[self.split_idx:].reset_index(drop=True)

                transformList = []
                transformList.append(transforms.RandomResizedCrop(256))
                transformList.append(transforms.RandomHorizontalFlip())
                transformList.append(transforms.ToTensor())
                transformList.append(transforms.Normalize(mean=0.5, std=0.5))      
                self.transform=transforms.Compose(transformList)

            
        elif mode == 'test':
            with open(os.path.join(self.data_path, "test_list.txt")) as file:
                image_list = [line.rstrip() for line in file] 
            df = df[df["image_index"].isin(image_list)]
            df["image_path"] = self.data_path+"/images/"+df["image_index"]
            df.drop(columns=["image_index"], inplace=True)
            self.final_image_df = df.sample(frac=1, random_state=45654).reset_index(drop=True)

            transformList = []
            transformList.append(transforms.Resize(256))
            transformList.append(transforms.ToTensor())
            transformList.append(transforms.Normalize(mean=0.5, std=0.5))  
            self.transform=transforms.Compose(transformList)

        
    def __len__(self):
        return len(self.final_image_df)

    def __getitem__(self, index):

        img_path = self.final_image_df.loc[index, 'image_path']
        image = PIL.Image.open(img_path)
        image = image.convert('L')
        image_transformed = self.transform(image).squeeze()
        return image_transformed, self.final_image_df.loc[index, [*self.labels, "no_finding"]].to_numpy(dtype=int)
        
  
if __name__ == "__main__":
    ds = ChestXRay_nih()
    print(ds[1][0].shape)