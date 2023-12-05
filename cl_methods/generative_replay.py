from typing import List, Callable, Tuple
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset, RandomSampler
import torch
from PIL import Image
import numpy as np

from cl_methods.base import CLMethod


class DatasetDummy(Dataset):
    def __init__(self, data, targets, transform=None):
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = None

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class GenerativeReplay(CLMethod):
    def get_data_for_task(
            self,
            sup_ds: Subset,
            unsup_ds: Subset,
            prev_tasks: List,
            samples_per_task: int,
            sample_generator: Callable[[int, List], Tuple[torch.Tensor, torch.Tensor]],
            filename: str = ""
    ):
        if len(prev_tasks) == 0:
            joined_sup_ds = sup_ds
            joined_unsup_ds = unsup_ds
        else:
            print("Preparing dataset for rehearsal...")
            to_generate = {task: samples_per_task for task in prev_tasks}
            generated_imgs = torch.tensor([])
            generated_labels = torch.tensor([])
            for task in to_generate.keys():
                while to_generate[task] > 0:
                    bs = min(self.args.sample_batch_size, to_generate[task])
                    imgs, labels = sample_generator(bs, [task for _ in range(bs)])
                    generated_imgs = torch.concat((generated_imgs, imgs))
                    generated_labels = torch.concat((generated_labels, labels))
                    to_generate[task] -= bs
            torch.save(generated_imgs, f"./data/cl/{filename}_imgs.pt")
            torch.save(generated_labels, f"./data/cl/{filename}_labels.pt")
            # generated_imgs = torch.load("./cifar10_images.pt")
            # generated_labels = torch.load("./cifar10_labels.pt")
            gen_sup_ds = DatasetDummy(
                (generated_imgs.permute((0, 2, 3, 1)).numpy() * 255).astype(np.uint8),
                generated_labels.numpy().astype(np.uint8),
                sup_ds.dataset.transform
            )
            joined_sup_ds = ConcatDataset([sup_ds, gen_sup_ds])

            gen_unsup_ds = DatasetDummy(
                (generated_imgs.permute((0, 2, 3, 1)).numpy() * 255).astype(np.uint8),
                generated_labels.numpy().astype(np.uint8),
                unsup_ds.dataset.transform
            )
            joined_unsup_ds = ConcatDataset([unsup_ds, gen_unsup_ds])

        labeled_dl = DataLoader(
            dataset=joined_sup_ds,
            sampler=RandomSampler(joined_sup_ds),
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.args.num_workers
        )
        unlabeled_dl = DataLoader(
            dataset=joined_unsup_ds,
            sampler=RandomSampler(joined_unsup_ds),
            batch_size=self.args.batch_size * 7,  # TODO
            shuffle=False,
            drop_last=True,
            num_workers=self.args.num_workers
        )
        return labeled_dl, unlabeled_dl
