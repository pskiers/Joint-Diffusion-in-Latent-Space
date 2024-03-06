import torch
import torchvision as tv
import random
from glob import glob
import os
from PIL import Image

class ChestXRay_nih_64(torch.utils.data.Dataset):
    def __init__(self, mode='train') -> None:
        super().__init__()
        self.data_path = "/home/jk/Joint-Diffusion-in-Latent-Space/chest_xray_nih"

        assert mode in ['val', 'train', 'test']

        if mode in ['val', 'train']:

            
            with open(os.path.join(self.data_path, "train_val_list.txt")) as file:
                image_list = [(os.path.join(self.data_path,'images',line.rstrip()),1) for line in file]
            random.Random(455455).shuffle(image_list)
            image_list = image_list[:30000] # TODO remove
            self.split_idx = int(0.0*len(image_list))
            
            if mode =='val':
                self.final_image_list = image_list[:self.split_idx].reset_index(drop=True)

                self.transform = tv.transforms.Compose([
                    tv.transforms.Resize((64,64)),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(mean=[.5], std=[.5])
                ])
            elif mode =='train':
                self.final_image_list = image_list[self.split_idx:].reset_index(drop=True)

                self.transform = tv.transforms.Compose([
                        tv.transforms.Resize((64,64)),
                        # tv.transforms.RandomCrop(32, padding=4),
                        tv.transforms.ToTensor(),
                        tv.transforms.Normalize(mean=[.5], std=[.5]),
                    ])
            
        elif mode == 'test':
            with open(os.path.join(self.data_path, "test_list.txt")) as file:
                image_list = [(os.path.join(self.data_path, "images", line.rstrip()),1) for line in file]
            
            random.Random(455455).shuffle(image_list)
            image_list = image_list[:5000] # TODO remove
            self.final_image_list = image_list

            self.transform = tv.transforms.Compose([
                    tv.transforms.Resize((64,64)),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(mean=[.5], std=[.5])
                ])


        
    def __len__(self):
        return len(self.final_image_list)

    def __getitem__(self, index):

        img_path = self.final_image_list[index][0]
        img = Image.open(img_path)
        img = img.convert('L')
        img_transformed = self.transform(img)
        return img_transformed.permute(1,2,0), self.final_image_list[index][1]
        
  

if __name__ == "__main__":
    ds = ChestXRay_nih_64()
    print(len(ds))