import torch
import torchvision as tv


class AdjustedCIFAR10(torch.utils.data.Dataset):
    def __init__(self, train=False) -> None:
        super().__init__()
        self.dataset = tv.datasets.CIFAR10(root="./data", train=train, download=True, transform=tv.transforms.ToTensor())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img.permute(1, 2, 0), label
    

class Pneumonia_dataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, mode, transform = None):
    
        self.data_path = dataroot
        self.mode = mode
        self.transform = transform

        if mode in ['val', 'train']:
            
            image_list = [(item, 1) for item in glob(os.path.join(self.data_path, 'train', 'PNEUMONIA')+'/*jpeg')] + \
                [(item, 0) for item in glob(os.path.join(self.data_path, 'train', 'NORMAL')+'/*jpeg')]
            random.Random(455455).shuffle(image_list)
            self.split_idx = int(0.1*len(image_list))
            
            if mode =='val':
                self.final_image_list = image_list[:self.split_idx]
            elif mode =='train':
                self.final_image_list = image_list[self.split_idx:]

        elif mode == 'test':
            image_list = [(item, 1) for item in glob(os.path.join(self.data_path, 'test', 'PNEUMONIA')+'/*jpeg')] + \
                [(item, 0) for item in glob(os.path.join(self.data_path, 'test', 'NORMAL')+'/*jpeg')]
            
            random.Random(455455).shuffle(image_list)
            self.final_image_list = image_list
        else:
            print('mode not imlemented')

        
    def __len__(self):
        return len(self.final_image_list)

    def __getitem__(self, index):

        img_path = self.final_image_list[index][0]
        img = Image.open(img_path)
        img = img.convert('L')
        img_transformed = self.transform(img)
        return img_transformed, self.final_image_list[index][1]
        
def Pneumonia_large(dataroot, skip_normalization=False, train_aug=False):
    print(dataroot)

    if skip_normalization:
        val_transform = transforms.Compose([
            CenterCropToSmallerEdge(100), #100 only to prevent error
            transforms.Resize((64,64)),
            transforms.ToTensor(),
        ])
    else:
        val_transform = transforms.Compose([
            CenterCropToSmallerEdge(100), #100 only to prevent error
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
        ])

    train_transform = val_transform

    if train_aug:
        train_transform = transforms.Compose([
            CenterCropToSmallerEdge(100), #100 only to prevent error
            transforms.Resize((64,64)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
        ])

    train_dataset = Pneumonia_dataset(dataroot=dataroot, mode='train', transform=train_transform)
    val_dataset = Pneumonia_dataset(dataroot=dataroot, mode='val', transform=val_transform)
    test_dataset = Pneumonia_dataset(dataroot=dataroot, mode='test', transform=val_transform)

    train_dataset.number_classes = 2
    val_dataset.number_classes = 2
    test_dataset.number_classes = 2
    return train_dataset, val_dataset, test_dataset, 64, 1

