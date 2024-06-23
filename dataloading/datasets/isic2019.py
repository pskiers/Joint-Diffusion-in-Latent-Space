from typing import Callable, Optional, Tuple
import pandas as pd
import os
from PIL import Image, UnidentifiedImageError

from .base import BaseDataset, Split


class ISIC2019Dataset(BaseDataset):
    def __init__(
        self,
        root: str = "data/isic-2019",
        split: Split = Split.TRAIN,
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        resize: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(root, split, download, transform, target_transform, **kwargs)
        self.img_folder_path = os.path.join(
            self.root, f"ISIC_2019_Training_Input_{resize}x{resize}"
        )
        if self.split == Split.TEST:
            df = pd.read_csv(os.path.join(self.root, "test_split.csv"))
        elif self.split == Split.TRAIN:
            df = pd.read_csv(os.path.join(self.root, "train_val_split.csv"))
        self.set_data(df["image_path"].values)
        targets = df[["mel", "nv", "bcc", "ak", "bkl", "df", "vasc", "scc"]].values
        targets = targets.argmax(axis=1)
        self.set_targets(targets)

    def __getitem__(self, index) -> Tuple:
        data = self.get_data()[index]
        data = Image.open(os.path.join(self.img_folder_path, str(data)))
        if self.transform is not None:
            data = self.transform(data)

        target = self.get_targets()[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def get_num_classes(self) -> int:
        return 8


if __name__ == "__main__":
    # prep dataset
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", "-r", type=str, required=True, help="path to config file"
    )
    args = parser.parse_args()
    path = os.path.join(args.root, "ISIC_2019_Training_Input")
    path_64 = os.path.join(args.root, "ISIC_2019_Training_Input_64x64")
    path_256 = os.path.join(args.root, "ISIC_2019_Training_Input_256x256")
    if not os.path.exists(path_64):
        os.makedirs(path_64)
    if not os.path.exists(path_256):
        os.makedirs(path_256)
    for img_path in tqdm(os.listdir(path)):
        try:
            image = Image.open(os.path.join(path, img_path))
            image_64 = image.resize((64, 64))
            image_64.save(os.path.join(path_64, img_path))
            image_256 = image.resize((256, 256))
            image_256.save(os.path.join(path_256, img_path))
        except UnidentifiedImageError:
            pass
