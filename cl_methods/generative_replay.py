from typing import List, Callable, Tuple, Optional, Union
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler, Dataset
import torch
from dataloading import BaseDataset, BaseTensorDataset


class GenerativeReplay:
    def __init__(
        self,
        train_bs: Union[int, List[int]],
        sample_bs: int = 200,
        dl_num_workers: int = 16,
    ):
        self.train_bs = train_bs
        self.sample_bs = sample_bs
        self.dl_num_workers = dl_num_workers

    def get_data_for_task(
        self,
        sup_ds: BaseDataset,
        unsup_ds: BaseDataset,
        prev_tasks: List,
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
        joined_sup_ds: Dataset
        joined_unsup_ds: Dataset
        if len(prev_tasks) == 0:
            joined_sup_ds = sup_ds
            joined_unsup_ds = unsup_ds
        else:
            print("Preparing dataset for rehearsal...")
            to_generate = {task: samples_per_task for task in prev_tasks}
            generated_imgs = torch.tensor([])
            generated_labels = torch.tensor([], dtype=torch.int64)
            if saved_samples is None or saved_labels is None:
                for task in to_generate.keys():
                    print(f"Sampling for label {task}")
                    while to_generate[task] > 0:
                        bs = min(self.sample_bs, to_generate[task])
                        imgs, labels = old_sample_generator(
                            bs, [task for _ in range(bs)]
                        )
                        generated_imgs = torch.concat((generated_imgs, imgs))
                        generated_labels = torch.concat((generated_labels, labels))
                        to_generate[task] -= len(labels)
                        print(f"{to_generate[task]} left for label {task}")
                if new_sample_generator is not None:
                    assert current_task is not None
                    to_generate = {task: samples_per_task for task in current_task}
                    for task in to_generate.keys():
                        print(f"Sampling for label {task}")
                        while to_generate[task] > 0:
                            bs = min(self.sample_bs, to_generate[task])
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
 
            gen_sup_ds = BaseTensorDataset(
                data=generated_imgs,
                targets=generated_labels,
                transform=sup_ds.transform
            )
            joined_sup_ds = (
                ConcatDataset([sup_ds, gen_sup_ds])
                if new_sample_generator is None
                else gen_sup_ds
            )

            gen_unsup_ds = BaseTensorDataset(
                data=generated_imgs, 
                targets=generated_labels,
                transform=unsup_ds.transform
            )
            joined_unsup_ds = (
                ConcatDataset([unsup_ds, gen_unsup_ds])
                if new_sample_generator is None
                else gen_unsup_ds
            )

        labeled_dl = DataLoader(
            dataset=joined_sup_ds,
            sampler=RandomSampler(joined_sup_ds),
            batch_size=self.train_bs if isinstance(self.train_bs, int) else self.train_bs[0],
            shuffle=False,
            drop_last=True,
            num_workers=self.dl_num_workers,
        )
        if new_sample_generator is None:
            assert isinstance(self.train_bs, list)
            unlabeled_dl = DataLoader(
                dataset=joined_unsup_ds,
                sampler=RandomSampler(joined_unsup_ds),
                batch_size=self.train_bs[1],
                shuffle=False,
                drop_last=True,
                num_workers=self.dl_num_workers,
            )
            return labeled_dl, unlabeled_dl
        return labeled_dl
