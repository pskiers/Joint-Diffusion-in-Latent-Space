from typing import List, Callable, Tuple, Optional
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

        if self.transform is not None:
            img = Image.fromarray(
                (img.permute((1, 2, 0)).numpy() * 255).astype(np.uint8)
            )
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
        mean: List[float],
        std: List[float],
        samples_per_task: int,
        old_sample_generator: Callable[[int, List], Tuple[torch.Tensor, torch.Tensor]],
        new_sample_generator: Optional[
            Callable[[int, List], Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        current_task: Optional[List] = None,
        filename: str = "",
        saved_samples: Optional[str] = None,
        saved_labels: Optional[str] = None,
    ):
        if len(prev_tasks) == 0:
            joined_sup_ds = sup_ds
            joined_unsup_ds = unsup_ds
        else:
            print("Preparing dataset for rehearsal...")
            to_generate = {task: samples_per_task for task in prev_tasks}
            generated_imgs = torch.tensor([])
            generated_labels = torch.tensor([]).type(torch.LongTensor)
            if saved_samples is None or saved_labels is None:
                for task in to_generate.keys():
                    print(f"Sampling for label {task}")
                    while to_generate[task] > 0:
                        bs = min(self.args.sample_batch_size, to_generate[task])
                        imgs, labels = old_sample_generator(
                            bs, [task for _ in range(bs)]
                        )
                        generated_imgs = torch.concat((generated_imgs, imgs))
                        generated_labels = torch.concat((generated_labels, labels))
                        to_generate[task] -= len(labels)
                        print(f"{to_generate[task]} left for label {task}")
                if new_sample_generator is not None:
                    to_generate = {task: samples_per_task for task in current_task}
                    for task in to_generate.keys():
                        print(f"Sampling for label {task}")
                        while to_generate[task] > 0:
                            bs = min(self.args.sample_batch_size, to_generate[task])
                            imgs, labels = new_sample_generator(
                                bs, [task for _ in range(bs)]
                            )
                            generated_imgs = torch.concat((generated_imgs, imgs))
                            generated_labels = torch.concat((generated_labels, labels))
                            to_generate[task] -= len(labels)
                            print(f"{to_generate[task]} left for label {task}")
                torch.save(generated_imgs, f"./data/cl/{filename}_imgs.pt")
                torch.save(generated_labels, f"./data/cl/{filename}_labels.pt")
            else:
                imgs = torch.load(saved_samples)
                labels = torch.load(saved_labels)
                generated_imgs = torch.concat((generated_imgs, imgs))
                generated_labels = torch.concat((generated_labels, labels))

            # generated_imgs = torch.load("./cifar100_images.pt") if filename == "cifar100_randaugment" else torch.load("./cifar10_images.pt")
            # generated_labels = torch.load("./cifar100_labels.pt").type(torch.LongTensor) if filename == "cifar100_randaugment" else torch.load("./cifar10_labels.pt").type(torch.LongTensor)

            from datasets.fixmatch_cifar import TransformRandAugmentSupervised  # TODO

            transform = TransformRandAugmentSupervised(mean=mean, std=std)  # TODO

            gen_sup_ds = DatasetDummy(
                generated_imgs,
                generated_labels,
                (
                    sup_ds.dataset.transform
                    if new_sample_generator is None
                    else transform
                ),  # TODO
            )
            joined_sup_ds = (
                ConcatDataset([sup_ds, gen_sup_ds])
                if new_sample_generator is None
                else gen_sup_ds
            )

            gen_unsup_ds = DatasetDummy(
                generated_imgs, generated_labels, unsup_ds.dataset.transform
            )
            joined_unsup_ds = (
                ConcatDataset([unsup_ds, gen_unsup_ds])
                if new_sample_generator is None
                else gen_unsup_ds
            )

        labeled_dl = DataLoader(
            dataset=joined_sup_ds,
            sampler=RandomSampler(joined_sup_ds),
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.args.num_workers,
        )
        if new_sample_generator is None:
            unlabeled_dl = DataLoader(
                dataset=joined_unsup_ds,
                sampler=RandomSampler(joined_unsup_ds),
                batch_size=self.args.batch_size * 7,  # TODO
                shuffle=False,
                drop_last=True,
                num_workers=self.args.num_workers,
            )
            return labeled_dl, unlabeled_dl
        return labeled_dl
