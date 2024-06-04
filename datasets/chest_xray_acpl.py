import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import io
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader

from .utils import SequentialDistributedSampler
from torch.utils.data import RandomSampler, DistributedSampler, Dataset
from torch.utils.data.sampler import SequentialSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets, models, transforms
# from utils.gcloud import download_chestxray_unzip

Labels = {
    "No Finding": 14,
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Consolidation": 2,
    "Edema": 3,
    "Effusion": 4,
    "Emphysema": 5,
    "Fibrosis": 6,
    "Hernia": 7,
    "Infiltration": 8,
    "Mass": 9,
    "Nodule": 10,
    "Pleural_Thickening": 11,
    "Pneumonia": 12,
    "Pneumothorax": 13,
}
mlb = MultiLabelBinarizer(classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])


class ChestACPLDataset(Dataset):
    def __init__(self, root_dir, mode, runtime=1, ratio=2) -> None:
        assert mode in ['labeled', 'unlabeled', 'anchor', 'test'], f"unsupported mode type {mode}"
        self.root_dir = root_dir
        self.mode = mode
        if self.mode == "labeled":
            transformList = []
            transformList.append(transforms.RandomResizedCrop(256, scale=(0.2, 1.0)))
            transformList.append(transforms.RandomHorizontalFlip())
            transformList.append(transforms.ToTensor())
            transformList.append(transforms.Normalize(0.5, 0.5)) #[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))  #adjusted to 1 channel    
            self.transform=transforms.Compose(transformList)
        else:
            transformList = []
            transformList.append(transforms.Resize(256))
            transformList.append(transforms.ToTensor())
            transformList.append(transforms.Normalize(0.5, 0.5)) #[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))  #adjusted to 1 channel
            self.transform=transforms.Compose(transformList)

        gr_path = os.path.join(root_dir, "Data_Entry_2017.csv")
        gr = pd.read_csv(gr_path, index_col=0)
        gr = gr.to_dict()["Finding Labels"]

        img_list = os.path.join(
            root_dir,
            "test_list.txt"
            if mode == "test"
            else "chest_xray_ssl_train_list_{}_{}.txt".format(ratio, runtime),
        )
        with open(img_list) as f:
            names = f.read().splitlines()
        self.labeled_imgs = np.asarray([x for x in names])

        all_img_list = os.path.join(root_dir, "train_val_list.txt")
        with open(all_img_list) as f:
            all_names = f.read().splitlines()

        labeled_gr = np.asarray([gr[i] for i in self.labeled_imgs])

        self.labeled_gr = np.zeros((labeled_gr.shape[0], 14), dtype=np.int32) #adjjusted to 14 classes
        for idx, i in enumerate(labeled_gr):
            target = i.split("|")
            binary_result = mlb.fit_transform([[Labels[i] for i in target]]).squeeze()
            self.labeled_gr[idx] = binary_result[:-1] #adjusted to 14 classes
        self.all_imgs = np.asarray([x for x in all_names])
        self.unlabeled_imgs = np.setdiff1d(self.all_imgs, self.labeled_imgs)
        unlabeled_gr = np.asarray([gr[i] for i in self.unlabeled_imgs])
        self.unlabeled_gr = np.zeros((unlabeled_gr.shape[0], 14), dtype=np.int32) #adjusted to 14 classes
        for idx, i in enumerate(unlabeled_gr):
            target = i.split("|")
            binary_result = mlb.fit_transform([[Labels[i] for i in target]]).squeeze()
            self.unlabeled_gr[idx] = binary_result[:-1] #adjusted to 14 classes

    def x_add_pl(self, pl, idxs):
        self.labeled_imgs = np.concatenate(
            (self.labeled_imgs, self.unlabeled_imgs[idxs])
        )
        self.labeled_gr = np.concatenate((self.labeled_gr, pl))

    def x_update_pl(self, idxs, args):
        print(self.labeled_imgs.shape)
        self.labeled_imgs = self.labeled_imgs[idxs]
        self.labeled_gr = self.labeled_gr[idxs]
        # mask = self.labeled_gr > (
        #     np.amax(self.labeled_gr, axis=1) - args.max_interval
        # ).reshape(-1, 1)
        # self.labeled_gr[mask] = 1
        # self.labeled_gr[~mask] = 0
        print(self.labeled_imgs.shape)

    def u_update_pl(self, idxs):
        print(self.unlabeled_imgs.shape)
        self.unlabeled_imgs = np.delete(self.unlabeled_imgs, idxs)
        self.unlabeled_gr = np.delete(self.unlabeled_gr, idxs, axis=0)
        print(self.unlabeled_imgs.shape)


    def __getitem__(self, item):

        if self.mode == "labeled" or self.mode == "anchor":
            img_path = os.path.join(self.root_dir, "images", self.labeled_imgs[item])
            input_path = float(self.labeled_imgs[item].split(".")[0].replace("_", "."))
            target = self.labeled_gr[item]
        elif self.mode == "unlabeled":
            img_path = os.path.join(self.root_dir, "images", self.unlabeled_imgs[item])
            input_path = float(
                self.unlabeled_imgs[item].split(".")[0].replace("_", ".")
            )
            target = self.unlabeled_gr[item]
        elif self.mode == "test":
            img_path = os.path.join(self.root_dir, "images", self.labeled_imgs[item])
            input_path = float(self.labeled_imgs[item].split(".")[0].replace("_", "."))
            target = self.labeled_gr[item]
            
        img = Image.fromarray(io.imread(img_path)).convert("L") #"RGB") #adjusted to 14 classes
        img_w = self.transform(img).squeeze() #adjusted to 14 classes
        return (img_w, target, item, input_path)

    def __len__(self):
        if self.mode == "labeled" or self.mode == "test" or self.mode == "anchor":
            return self.labeled_imgs.shape[0]
        elif self.mode == "unlabeled":
            return self.unlabeled_imgs.shape[0]


class ChestACPLDataloader:
    def __init__(
        self,
        batch_size=128,
        num_workers=8,
        img_resize=512,
        training_platform='local_sano',
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_resize = img_resize

        if training_platform=="local_sano":
            self.root_dir = "/home/jk/Joint-Diffusion-in-Latent-Space/chest_xray_nih"  
        elif training_platform=="de":
            self.root_dir = "/data/jan_dubinski/Joint-Diffusion-in-Latent-Space/sano_machines/data"
        else:
            raise NotImplemented("no training platform for ACPL")

    def run(self, mode, dataset=None, ratio=2, runtime=1):

        if dataset:
            all_dataset = dataset
        else:
            all_dataset = ChestACPLDataset(
                root_dir=self.root_dir,
                mode=mode,
                ratio=ratio,
                runtime=runtime,
            )
        batch_size = (
            (self.batch_size * 1) if mode == "test"
            else (self.batch_size * 1) if mode == "unlabeled" or mode == "anchor"
            else self.batch_size
        )
        sampler = (
            RandomSampler(all_dataset)
            # if mode == "labeled"
            # else SequentialSampler(all_dataset)
        )
        loader = DataLoader(
            dataset=all_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True if mode == "labeled" else False,
        )

        return loader, all_dataset, sampler

    # def update(self,pl):


if __name__ == "__main__":
    ds = ChestACPLDataset(root_dir = "/home/jk/Joint-Diffusion-in-Latent-Space/chest_xray_nih", mode='labeled')
    print(len(ds), ds[0])