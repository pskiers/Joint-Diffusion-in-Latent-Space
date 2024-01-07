import torch
import torchvision as tv
import random
from glob import glob
import os
from PIL import Image

class ChestXRay(torch.utils.data.Dataset):
    def __init__(self, mode='train') -> None:
        super().__init__()
        self.data_path = "/home/jk/Joint-Diffusion-in-Latent-Space/chest_xray"

        assert mode in ['val', 'train', 'test']
        if mode in ['val', 'train']:
            
            image_list = [(item, 1) for item in glob(os.path.join(self.data_path, 'train', 'PNEUMONIA')+'/*jpeg')] + \
                [(item, 0) for item in glob(os.path.join(self.data_path, 'train', 'NORMAL')+'/*jpeg')]
            random.Random(455455).shuffle(image_list)
            self.split_idx = int(0.1*len(image_list))
            
            if mode =='val':
                self.final_image_list = image_list[:self.split_idx]

                self.transform = tv.transforms.Compose([
                    tv.transforms.Resize((256,256)),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(mean=[.5], std=[.5])
                ])
            elif mode =='train':
                self.final_image_list = image_list[self.split_idx:]

                self.transform = tv.transforms.Compose([
                        tv.transforms.Resize((256,256)),
                        # tv.transforms.RandomCrop(32, padding=4),
                        tv.transforms.ToTensor(),
                        tv.transforms.Normalize(mean=[.5], std=[.5]),
                    ])
            
        elif mode == 'test':
            image_list = [(item, 1) for item in glob(os.path.join(self.data_path, 'test', 'PNEUMONIA')+'/*jpeg')] + \
                [(item, 0) for item in glob(os.path.join(self.data_path, 'test', 'NORMAL')+'/*jpeg')]
            
            random.Random(455455).shuffle(image_list)
            self.final_image_list = image_list

            self.transform = tv.transforms.Compose([
                    tv.transforms.Resize((256, 256)),
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
    ds = ChestXRay()
    print(ds[1][0].shape)