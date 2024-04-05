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


class ISIC2019(torch.utils.data.Dataset):
    def __init__(self, mode='train', training_platform: str = 'local_sano', extend_with_test = False) -> None:
        super().__init__()

        assert training_platform in ['plgrid', 'local_sano', "de"]
        if training_platform=='plgrid':
            self.data_path = f"{os.environ['SCRATCH']}/isic2019"
        elif training_platform=='local_sano':
            self.data_path = "/home/jk/Joint-Diffusion-in-Latent-Space/isic2019"
        elif training_platform=='de':
            self.data_path = "/data/jan_dubinski/Joint-Diffusion-in-Latent-Space/sano_machines/isic2019"
        
        assert mode in ['val', 'train', 'test']
        df = pd.read_csv(os.path.join(self.data_path, "ISIC_2019_Training_GroundTruth.csv"))
        self.split_idx = self.split_idx_val = int(0.0*len(df))

        # # PART USED ONLY FOR SPLIT to CSV
        # df.rename(columns=lambda x: x.replace(" ", "_").lower(), inplace=True)
        # df["image_path"] = df["image"]+'.jpg'
        # df.drop(columns=["image"], inplace=True)
        # df = df.sample(frac=1, random_state=45654).reset_index(drop=True) 
        # self.split_idx_test = int(0.2*len(df))
        # self.split_idx_val = self.split_idx_test + int(0.1*len(df))
        # self.final_image_df = df[:self.split_idx_test].reset_index(drop=True)
        # self.final_image_df.to_csv(os.path.join(self.data_path, "test_split.csv"))
        # self.final_image_df = df[self.split_idx_test:].reset_index(drop=True)
        # self.final_image_df.to_csv(os.path.join(self.data_path, "train_val_split.csv"))
        
        if mode in ['val', 'train']:
            self.final_image_df = pd.read_csv(os.path.join(self.data_path, "train_val_split.csv"), index_col=0)
            self.final_image_df["image_path"] = self.data_path+"/images/"+ self.final_image_df["image_path"]
            self.labels = self.final_image_df.columns[~self.final_image_df.columns.isin(['image_path', "unk"])]

            if mode =='val':
                self.final_image_df = self.final_image_df[:self.split_idx].reset_index(drop=True)

                transformList = []
                transformList.append(transforms.Resize(256))
                transformList.append(transforms.CenterCrop(256))
                transformList.append(transforms.ToTensor())
                transformList.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))  
                self.transform=transforms.Compose(transformList)

            elif mode =='train':
                self.final_image_df = self.final_image_df[self.split_idx:].reset_index(drop=True)

                transformList = []
                transformList.append(transforms.RandomResizedCrop(256, (0.5, 1.0)))
                transformList.append(transforms.RandomHorizontalFlip())
                transformList.append(transforms.ToTensor())
                transformList.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))      
                self.transform=transforms.Compose(transformList)

                if extend_with_test:
                    additional_imgs = os.path.join(self.data_path, "additional_test_images_no_gt")
                    img_list =[os.path.join(additional_imgs, f) for f in os.listdir(additional_imgs) if '.jpg' in f]
                    add_df = pd.DataFrame(np.ones((len(img_list), 9))*(-1))
                    add_df['image_path'] = img_list
                    add_df.columns = self.final_image_df.columns
                    self.final_image_df = pd.concat((self.final_image_df, add_df), axis=0, ignore_index=True).reset_index(drop=True)

        elif mode == 'test':
            self.final_image_df = pd.read_csv(os.path.join(self.data_path, "test_split.csv"), index_col=0).reset_index(drop=True)
            self.final_image_df["image_path"] = self.data_path+"/images/"+ self.final_image_df["image_path"]
            self.labels = self.final_image_df.columns[~self.final_image_df.columns.isin(['image_path', "unk"])]

            transformList = []
            transformList.append(transforms.Resize(256))
            transformList.append(transforms.CenterCrop(256))
            transformList.append(transforms.ToTensor())
            transformList.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))  
            self.transform=transforms.Compose(transformList)

        
    def __len__(self):
        return len(self.final_image_df)

    def __getitem__(self, index):

        img_path = self.final_image_df.loc[index, 'image_path']
        image = PIL.Image.open(img_path)
        image_transformed = self.transform(image).squeeze()
        return image_transformed.permute(1,2,0), self.final_image_df.loc[index, [*self.labels, "unk"]].to_numpy(dtype=int)
        
  
if __name__ == "__main__":
    ds = ISIC2019(mode="train", extend_with_test=True)
    print(len(ds))
    print(ds[24436][1], ds[0][0].shape)

    # ds = ISIC2019(mode="val")
    # print(len(ds))
    # print(ds[0][1], ds[0][0].shape)

    ds = ISIC2019(mode="test")
    print(len(ds))
    print(ds[2436][1], ds[2][0].shape)
