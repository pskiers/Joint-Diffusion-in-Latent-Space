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
    def __init__(self, mode='train', training_platform: str = 'local_sano',  img_size=256, min_augmentation_ratio: int = 0.8, auto_augment=False) -> None:
        super().__init__()
        self.auto_augment = auto_augment if mode=='train' else False

        assert training_platform in ['plgrid', 'local_sano', "de"]
        if training_platform =='plgrid':
            data_path = f"{os.environ['SCRATCH']}/chest_xray_nih"
        elif training_platform=='local_sano':
            data_path = "/home/jk/Joint-Diffusion-in-Latent-Space/chest_xray_nih"
        elif training_platform =='local_pawel':
            data_path = "/home/pawel/studia/Joint-Diffusion-in-Latent-Space/data/chest_xray"
        elif training_platform =='de':
            data_path = "/data/jan_dubinski/Joint-Diffusion-in-Latent-Space/sano_machines/data"

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
                    A.Resize(img_size, img_size),
                    A.Normalize(mean=0.5, std=0.5),
                ])
            elif mode =='train':
                self.final_image_df = df[self.split_idx:]

                if self.auto_augment:
                    self.transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize(0.5, 0.5),

                    ])
                else:
                    self.transform = A.Compose([
                        A.RandomResizedCrop(width=img_size, height=img_size, scale=(min_augmentation_ratio, 1.0), ratio=(0.75, 1.33)),
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
                    A.Resize(img_size,img_size),
                    A.Normalize(mean=0.5, std=0.5),
                ])

    def __len__(self):
        return len(self.final_image_df)

    def __getitem__(self, index):

        img_path = self.final_image_df.loc[index, 'image_path']
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

        return image_transformed, self.final_image_df.loc[index, [*self.labels, "no_finding"]].to_numpy(dtype=int)


if __name__ == "__main__":
    ds = ChestXRay_nih(mode="train", training_platform="local_sano", img_size=256, min_augmentation_ratio=0.8, auto_augment=False)
    print(ds[1][0].shape)
