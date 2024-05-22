from abc import ABC, abstractmethod
from typing import List, Callable, Tuple, Optional, Union
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler, Dataset
import torch
import torchvision.transforms as transforms
from dataloading import BaseDataset, BaseTensorDataset


class Replay(ABC):
    def __init__(
        self,
        train_bs: Union[int, List[int]],
        sample_bs: int = 200,
        dl_num_workers: int = 16,
    ):
        self.train_bs = train_bs
        self.sample_bs = sample_bs
        self.dl_num_workers = dl_num_workers

    @abstractmethod
    def get_data_for_task(
        self,
        sup_ds: BaseDataset,
        unsup_ds: BaseDataset,
        prev_tasks: List,
        samples_per_task: int,
        old_sample_generator: Callable[[int, List], Tuple[torch.Tensor, torch.Tensor]],
        new_sample_generator: Optional[Callable[[int, List], Tuple[torch.Tensor, torch.Tensor]]] = None,
        current_task: Optional[List] = None,
        filename: str = "",
        saved_samples: Optional[str] = None,
        saved_labels: Optional[str] = None,
    ):
        """
        Get data for task
        """


class NoReplaySSL(Replay):
    """
    Class for training model on a single ssl task without any generative replay
    """

    def get_data_for_task(
        self,
        sup_ds: BaseDataset,
        unsup_ds: BaseDataset,
        prev_tasks: List,
        samples_per_task: int,
        old_sample_generator: Callable[[int, List], Tuple[torch.Tensor, torch.Tensor]],
        new_sample_generator: Optional[Callable[[int, List], Tuple[torch.Tensor, torch.Tensor]]] = None,
        current_task: Optional[List] = None,
        filename: str = "",
        saved_samples: Optional[str] = None,
        saved_labels: Optional[str] = None,
    ):
        assert isinstance(self.train_bs, list)
        labeled_bs = self.train_bs[0]
        labeled_dl = DataLoader(
            dataset=sup_ds,
            sampler=RandomSampler(sup_ds, num_samples=max(len(sup_ds), labeled_bs * 500)),
            batch_size=labeled_bs,
            shuffle=False,
            drop_last=True,
            num_workers=self.dl_num_workers,
        )
        unlabeled_dl = DataLoader(
            dataset=unsup_ds,
            sampler=RandomSampler(
                unsup_ds,
                num_samples=max(len(unsup_ds), self.train_bs[1] * 500),
            ),
            batch_size=self.train_bs[1],
            shuffle=False,
            drop_last=True,
            num_workers=self.dl_num_workers,
        )
        return labeled_dl, unlabeled_dl


class NoReplaySupervised(Replay):
    """
    Class for training model on a single supervised task without any generative replay
    """

    def get_data_for_task(
        self,
        sup_ds: BaseDataset,
        unsup_ds: BaseDataset,
        prev_tasks: List,
        samples_per_task: int,
        old_sample_generator: Callable[[int, List], Tuple[torch.Tensor, torch.Tensor]],
        new_sample_generator: Optional[Callable[[int, List], Tuple[torch.Tensor, torch.Tensor]]] = None,
        current_task: Optional[List] = None,
        filename: str = "",
        saved_samples: Optional[str] = None,
        saved_labels: Optional[str] = None,
    ):
        assert isinstance(self.train_bs, list)
        labeled_bs = self.train_bs[0]
        labeled_dl = DataLoader(
            dataset=sup_ds,
            sampler=RandomSampler(sup_ds, num_samples=max(len(sup_ds), labeled_bs * 500)),
            batch_size=labeled_bs,
            shuffle=False,
            drop_last=True,
            num_workers=self.dl_num_workers,
        )
        return labeled_dl


class ConditionalOnlyGenerativeReplay(Replay):
    """
    Generative replay with conditional sampling - generations only, no joining with new dataset
    """

    def get_data_for_task(
        self,
        sup_ds: BaseDataset,
        unsup_ds: BaseDataset,
        prev_tasks: List,
        samples_per_task: int,
        old_sample_generator: Callable[[int, List], Tuple[torch.Tensor, torch.Tensor]],
        new_sample_generator: Optional[Callable[[int, List], Tuple[torch.Tensor, torch.Tensor]]] = None,
        current_task: Optional[List] = None,
        filename: str = "",
        saved_samples: Optional[str] = None,
        saved_labels: Optional[str] = None,
    ):
        assert isinstance(self.train_bs, list)
        assert new_sample_generator is not None

        generated_imgs = torch.tensor([])
        generated_labels = torch.tensor([], dtype=torch.int64)
        print("Preparing dataset for rehearsal...")

        if saved_samples is None or saved_labels is None:
            # samples for old tasks
            to_generate = {task: samples_per_task for task in prev_tasks}
            for task in to_generate.keys():
                print(f"Sampling for label {task}")
                while to_generate[task] > 0:
                    bs = min(self.sample_bs, to_generate[task])
                    imgs, labels = old_sample_generator(bs, [task for _ in range(bs)])
                    generated_imgs = torch.concat((generated_imgs, imgs))
                    generated_labels = torch.concat((generated_labels, labels))
                    to_generate[task] -= len(labels)
                    print(f"{to_generate[task]} left for label {task}")
            assert current_task is not None

            # samples for new task
            to_generate = {task: samples_per_task for task in current_task}
            for task in to_generate.keys():
                print(f"Sampling for label {task}")
                while to_generate[task] > 0:
                    bs = min(self.sample_bs, to_generate[task])
                    imgs, labels = new_sample_generator(bs, [task for _ in range(bs)])
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

        gen_ds = BaseTensorDataset(
            data=generated_imgs,
            targets=generated_labels.numpy(),
            transform=sup_ds.transform,
        )

        labeled_bs = self.train_bs[0]
        labeled_dl = DataLoader(
            dataset=gen_ds,
            sampler=RandomSampler(gen_ds, num_samples=max(len(gen_ds), labeled_bs * 500)),
            batch_size=labeled_bs,
            shuffle=False,
            drop_last=True,
            num_workers=self.dl_num_workers,
        )
        return labeled_dl


class ConditionalGenerativeCombinedReplay(Replay):
    """
    Generative replay with conditional sampling (only for supervised)- generations dataset is joint with the new dataset
    """

    def get_data_for_task(
        self,
        sup_ds: BaseDataset,
        unsup_ds: BaseDataset,
        prev_tasks: List,
        samples_per_task: int,
        old_sample_generator: Callable[[int, List], Tuple[torch.Tensor, torch.Tensor]],
        new_sample_generator: Optional[Callable[[int, List], Tuple[torch.Tensor, torch.Tensor]]] = None,
        current_task: Optional[List] = None,
        filename: str = "",
        saved_samples: Optional[str] = None,
        saved_labels: Optional[str] = None,
    ):
        assert isinstance(self.train_bs, list)

        generated_imgs = torch.tensor([])
        generated_labels = torch.tensor([], dtype=torch.int64)
        print("Preparing dataset for rehearsal...")

        if saved_samples is None or saved_labels is None:
            # samples for old tasks
            to_generate = {task: samples_per_task for task in prev_tasks}
            for task in to_generate.keys():
                print(f"Sampling for label {task}")
                while to_generate[task] > 0:
                    bs = min(self.sample_bs, to_generate[task])
                    imgs, labels = old_sample_generator(bs, [task for _ in range(bs)])
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

        gen_ds = BaseTensorDataset(
            data=generated_imgs,
            targets=generated_labels.numpy(),
            transform=sup_ds.transform,
        )
        joined_sup_ds = ConcatDataset([sup_ds, gen_ds])
        labeled_bs = self.train_bs[0]
        labeled_dl = DataLoader(
            dataset=joined_sup_ds,
            sampler=RandomSampler(joined_sup_ds, num_samples=max(len(joined_sup_ds), labeled_bs * 500)),
            batch_size=labeled_bs,
            shuffle=False,
            drop_last=True,
            num_workers=self.dl_num_workers,
        )
        return labeled_dl


class UnconditionalOnlyGenerativeReplay(Replay):
    """
    Generative replay with unconditional sampling - generations only, no joining with new dataset
    """

    def get_data_for_task(
        self,
        sup_ds: BaseDataset,
        unsup_ds: BaseDataset,
        prev_tasks: List,
        samples_per_task: int,
        old_sample_generator: Callable[[int, List], Tuple[torch.Tensor, torch.Tensor]],
        new_sample_generator: Optional[Callable[[int, List], Tuple[torch.Tensor, torch.Tensor]]] = None,
        current_task: Optional[List] = None,
        filename: str = "",
        saved_samples: Optional[str] = None,
        saved_labels: Optional[str] = None,
    ):
        assert isinstance(self.train_bs, list)
        # assert new_sample_generator is not None

        generated_imgs = torch.tensor([])
        generated_labels = torch.tensor([], dtype=torch.int64)
        print("Preparing dataset for rehearsal...")

        if saved_samples is None or saved_labels is None:
            # samples for old tasks
            to_generate = {task: samples_per_task for task in prev_tasks}
            generated = 0
            while any([val > 0 for val in to_generate.values()]):
                imgs, labels = old_sample_generator(self.sample_bs, None)
                generated += len(imgs)
                for task, sampl_left in to_generate.items():
                    mask = labels == task if len(labels.shape) == 1 else labels.argmax(dim=-1) == task
                    task_imgs = imgs[mask][:sampl_left]
                    task_labels = labels[mask][:sampl_left]
                    generated_imgs = torch.concat((generated_imgs, task_imgs))
                    generated_labels = torch.concat((generated_labels, task_labels))
                    to_generate[task] -= len(task_labels)
                    print(f"{to_generate[task]} left for label {task}")
                if generated > len(prev_tasks) * samples_per_task * 2:
                    transform = transforms.Compose(
                        [
                            transforms.ToPILImage(),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomResizedCrop(
                                size=generated_imgs.shape[2], scale=[0.90, 1.0], antialias=True
                            ),
                            transforms.ToTensor(),
                        ]
                    )
                    for task, sampl_left in to_generate.items():
                        while sampl_left > 0:
                            task_imgs = generated_imgs[generated_labels == task][:sampl_left]
                            for i, img in enumerate(task_imgs):
                                task_imgs[i] = transform(img)
                            task_labels = generated_labels[generated_labels == task][:sampl_left]
                            generated_imgs = torch.concat((generated_imgs, task_imgs))
                            generated_labels = torch.concat((generated_labels, task_labels))
                            sampl_left -= len(task_labels)
                            print(f"{sampl_left} left for label {task}")
                    break

            # samples for new task
            assert current_task is not None
            to_generate = {task: samples_per_task for task in current_task}
            generated = 0
            while any([val > 0 for val in to_generate.values()]):
                imgs, labels = new_sample_generator(self.sample_bs, None)
                generated += len(imgs)
                for task, sampl_left in to_generate.items():
                    mask = labels == task if len(labels.shape) == 1 else labels.argmax(dim=-1) == task
                    task_imgs = imgs[mask][:sampl_left]
                    task_labels = labels[mask][:sampl_left]
                    generated_imgs = torch.concat((generated_imgs, task_imgs))
                    generated_labels = torch.concat((generated_labels, task_labels))
                    to_generate[task] -= len(task_labels)
                    print(f"{to_generate[task]} left for label {task}")
                if generated > len(current_task) * samples_per_task * 2:
                    transform = transforms.Compose(
                        [
                            transforms.ToPILImage(),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomResizedCrop(
                                size=generated_imgs.shape[2], scale=[0.85, 1.0], antialias=True
                            ),
                            transforms.ToTensor(),
                        ]
                    )
                    for task, sampl_left in to_generate.items():
                        while sampl_left > 0:
                            task_imgs = generated_imgs[generated_labels == task][:sampl_left]
                            for i, img in enumerate(task_imgs):
                                task_imgs[i] = transform(img)
                            task_labels = generated_labels[generated_labels == task][:sampl_left]
                            generated_imgs = torch.concat((generated_imgs, task_imgs))
                            generated_labels = torch.concat((generated_labels, task_labels))
                            sampl_left -= len(task_labels)
                            print(f"{sampl_left} left for label {task}")
                    break

            torch.save(generated_imgs, f"./data/cl/{filename}_imgs.pt")
            torch.save(generated_labels, f"./data/cl/{filename}_labels.pt")
        else:
            imgs = torch.load(saved_samples)
            labels = torch.load(saved_labels)
            generated_imgs = torch.concat((generated_imgs, imgs))
            generated_labels = torch.concat((generated_labels, labels))

        gen_ds = BaseTensorDataset(
            data=generated_imgs,
            targets=generated_labels.numpy(),
            transform=sup_ds.transform,
        )

        labeled_bs = self.train_bs[0]
        labeled_dl = DataLoader(
            dataset=gen_ds,
            sampler=RandomSampler(gen_ds, num_samples=max(len(gen_ds), labeled_bs * 500)),
            batch_size=labeled_bs,
            shuffle=False,
            drop_last=True,
            num_workers=self.dl_num_workers,
        )
        return labeled_dl


class UnconditionalGenerativeCombinedReplay(Replay):
    """
    Generative replay with unconditional sampling (only for supervised)- generations dataset is joint with the new dataset
    """

    def get_data_for_task(
        self,
        sup_ds: BaseDataset,
        unsup_ds: BaseDataset,
        prev_tasks: List,
        samples_per_task: int,
        old_sample_generator: Callable[[int, List], Tuple[torch.Tensor, torch.Tensor]],
        new_sample_generator: Optional[Callable[[int, List], Tuple[torch.Tensor, torch.Tensor]]] = None,
        current_task: Optional[List] = None,
        filename: str = "",
        saved_samples: Optional[str] = None,
        saved_labels: Optional[str] = None,
    ):
        assert isinstance(self.train_bs, list)

        generated_imgs = torch.tensor([])
        generated_labels = torch.tensor([], dtype=torch.int64)
        print("Preparing dataset for rehearsal...")

        if saved_samples is None or saved_labels is None:
            # samples for old tasks
            to_generate = {task: samples_per_task for task in prev_tasks}
            generated = 0
            while any([val > 0 for val in to_generate.values()]):
                imgs, labels = old_sample_generator(self.sample_bs, None)
                generated += len(imgs)
                for task, sampl_left in to_generate.items():
                    task_imgs = imgs[labels == task][:sampl_left]
                    task_labels = labels[labels == task][:sampl_left]
                    generated_imgs = torch.concat((generated_imgs, task_imgs))
                    generated_labels = torch.concat((generated_labels, task_labels))
                    to_generate[task] -= len(task_labels)
                    print(f"{to_generate[task]} left for label {task}")
                if generated > len(prev_tasks) * samples_per_task * 2:
                    transform = transforms.Compose(
                        [
                            transforms.ToPILImage(),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomResizedCrop(
                                size=generated_imgs.shape[2], scale=[0.85, 1.0], antialias=True
                            ),
                            transforms.ToTensor(),
                        ]
                    )
                    for task, sampl_left in to_generate.items():
                        while sampl_left > 0:
                            task_imgs = generated_imgs[generated_labels == task][:sampl_left]
                            for i, img in enumerate(task_imgs):
                                task_imgs[i] = transform(img)
                            task_labels = generated_labels[generated_labels == task][:sampl_left]
                            generated_imgs = torch.concat((generated_imgs, task_imgs))
                            generated_labels = torch.concat((generated_labels, task_labels))
                            sampl_left -= len(task_labels)
                            print(f"{sampl_left} left for label {task}")
                    break
            torch.save(generated_imgs, f"./data/cl/{filename}_imgs.pt")
            torch.save(generated_labels, f"./data/cl/{filename}_labels.pt")
        else:
            imgs = torch.load(saved_samples)
            labels = torch.load(saved_labels)
            generated_imgs = torch.concat((generated_imgs, imgs))
            generated_labels = torch.concat((generated_labels, labels))

        gen_ds = BaseTensorDataset(
            data=generated_imgs,
            targets=generated_labels.numpy(),
            transform=sup_ds.transform,
        )
        joined_sup_ds = ConcatDataset([sup_ds, gen_ds])
        labeled_bs = self.train_bs[0]
        labeled_dl = DataLoader(
            dataset=joined_sup_ds,
            sampler=RandomSampler(joined_sup_ds, num_samples=max(len(joined_sup_ds), labeled_bs * 500)),
            batch_size=labeled_bs,
            shuffle=False,
            drop_last=True,
            num_workers=self.dl_num_workers,
        )
        return labeled_dl


def get_replay(name: str) -> type:
    name_to_replay = {
        "one_ssl_task": NoReplaySSL,
        "one_supervised_task": NoReplaySupervised,
        "conditional_replay_only": ConditionalOnlyGenerativeReplay,
        "conditional_replay_and_new_dataset": ConditionalGenerativeCombinedReplay,
        "unconditional_replay_only": UnconditionalOnlyGenerativeReplay,
        "unconditional_replay_and_new_dataset": UnconditionalGenerativeCombinedReplay,
    }
    return name_to_replay[name]
