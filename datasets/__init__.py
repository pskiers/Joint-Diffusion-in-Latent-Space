from typing import Tuple, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torchvision import datasets

from .cifar10 import AdjustedCIFAR10
from .cifar100 import AdjustedCIFAR100
from .chest_xray_nih import ChestXRay_nih
from .chest_xray_nih_densenet import ChestXRay_nih_densenet
from .chest_xray_nih_densenet_2perc import ChestXRay_nih_densenet_2perc
from .chest_xray_nih_ssl import ChestXRay_nih_ssl
from .arch.chest_xray import ChestXRay
from .chest_xray_nih_patches import ChestXRay_nih_patches
from .chest_xray_nih_densenet_patches import ChestXRay_nih_densenet_patches
from .chest_xray_nih_bbox import ChestXRay_nih_bbox
from .chest_xray_acpl import ChestACPLDataloader, ChestACPLDataset
from .arch.chest_xray_nih_64 import ChestXRay_nih_64
from .isic2019 import ISIC2019
from .mnist import AdjustedMNIST
from .cleba import AdjustedCelbA
from .fashionMNIST import AdjustedFashionMNIST
from .svhn import AdjustedSVHN
from .gtsrb import GTSRB
from .utils import equal_labels_random_split, cl_class_split
from .fixmatch_cifar import DATASET_GETTERS
from pytorch_multilabel_balanced_sampler.samplers import RandomClassSampler, ClassCycleSampler, LeastSampledClassSampler
from .acpl_plmodule import ACPLDataModule
import warnings

@dataclass
class RandAugmentArgs:
    num_labeled: int = 1000
    num_classes: int = 10
    expand_labels: bool = True
    batch_size: int = 64
    eval_step: int = 1024


def get_dataloaders(name: str,
                    train_batches: Tuple[int],
                    val_batch: int,
                    num_workers: int,
                    num_labeled: Optional[int] = None,
                    pin_memory: bool = False,
                    persistent_workers: bool = False,
                    training_platform: str = 'plgrid',
                    ):
    
    if name=='chest_xray_nih':
        train_ds = ChestXRay_nih(mode='train', training_platform = training_platform)
        val_ds = ChestXRay_nih(mode='val', training_platform = training_platform)
        test_ds = ChestXRay_nih(mode='test', training_platform = training_platform)
        return train_test_val_dl(
            train_ds, val_ds, test_ds, train_batches, val_batch, num_workers, 
            pin_memory=pin_memory, persistent_workers=persistent_workers)
    elif name=='chest_xray_nih_encoder':
        train_ds = ChestXRay_nih(mode='train', training_platform = training_platform, val_split_ratio=0)
        val_ds = ChestXRay_nih(mode='test', training_platform = training_platform)
        test_ds = ChestXRay_nih(mode='test', training_platform = training_platform)
        return train_test_val_dl(
            train_ds, val_ds, test_ds, train_batches, val_batch, num_workers, 
            pin_memory=pin_memory, persistent_workers=persistent_workers)
    
    elif name=='chest_xray_nih_2perc':
        train_ds = ChestXRay_nih_ssl(mode='train', 
                                    training_platform = training_platform, labeled = True)
        val_ds = ChestXRay_nih_ssl(mode='val', training_platform = training_platform, labeled = True)
        test_ds = ChestXRay_nih(mode='test', training_platform = training_platform)
        return train_test_val_dl(
            train_ds, val_ds, test_ds,train_batches, val_batch, num_workers, 
            pin_memory=pin_memory, persistent_workers=persistent_workers)
    
    elif name=='chest_xray_nih_ssl':
            #Not optimized - we repeat the same operation 3 times. But leave it for now 
            labeled_ds = ChestXRay_nih_ssl(mode='train', 
                                training_platform = training_platform,
                                labeled = True)
            unlabeled_ds = ChestXRay_nih_ssl(mode='train', 
                                training_platform = training_platform, 
                                labeled = False)
            val_ds = ChestXRay_nih_ssl(mode='val', 
                                training_platform = training_platform, 
                                labeled = True)
            test_ds = ChestXRay_nih(mode='test', training_platform = training_platform)
            return ssl_basic_dl(
                 labeled_ds, unlabeled_ds, val_ds, test_ds, 
                 train_batches[0], val_batch, num_workers)
    
    elif name=='chest_xray_nih_patches':
            ds = ChestXRay_nih_patches(training_platform = training_platform)
            dl = torch.utils.data.DataLoader(
                ds,
                batch_size=val_batch,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers = persistent_workers
            )
            return dl
    
    elif name=='chest_xray_nih_densenet':
        train_ds = ChestXRay_nih_densenet(mode='train', 
                                       training_platform = training_platform)
        val_ds = ChestXRay_nih_densenet(mode='val', training_platform = training_platform)
        test_ds = ChestXRay_nih_densenet(mode='test', training_platform = training_platform)
        return train_test_val_dl(
            train_ds, val_ds, test_ds, train_batches, val_batch, num_workers, 
            pin_memory=pin_memory, persistent_workers=persistent_workers)
    elif name=='chest_xray_nih_densenet_2perc':
        train_ds = ChestXRay_nih_densenet_2perc(mode='train', 
                                       training_platform = training_platform)
        val_ds = ChestXRay_nih_densenet_2perc(mode='val', training_platform = training_platform)
        test_ds = ChestXRay_nih_densenet_2perc(mode='test', training_platform = training_platform)
        return train_test_val_dl(
            train_ds, val_ds, test_ds, train_batches, val_batch, num_workers, 
            pin_memory=pin_memory, persistent_workers=persistent_workers)
    elif name=='chest_xray_nih_densenet_patches':
            ds = ChestXRay_nih_densenet_patches(training_platform = training_platform)
            dl = torch.utils.data.DataLoader(
                ds,
                batch_size=val_batch,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers = persistent_workers
            )
            return dl
    elif name=='chest_xray_acpl':
        label_ratio=2
        runtime=3
        warnings.warn('&&&&&&&&&&&&&&&&&& ACPL DATASETS - REMEMBER WE HAVE LABEL RATIO AND RUNTIME HARDCODED AND MOCKED IN DIFFERENT PLACES!!!!!!!!!')
        loader = ChestACPLDataloader(
            batch_size=16,
            num_workers=num_workers,
            training_platform='local_sano',
        )
        (test_loader, test_dataset, test_sampler) = loader.run(
            "test",
            ratio=label_ratio,
            runtime=runtime,
        )
        
        (label_loader1, label_dataset, label_sampler,) = loader.run(
            "labeled",
            ratio=label_ratio,
            runtime=runtime,
        )
        
        (anchor_loader, anchor_dataset, anchor_sampler,) = loader.run(
            "anchor",
            ratio=label_ratio,
            runtime=runtime,
        )

        if label_ratio != 100:
            (unlabel_loader, unlabel_dataset, unlabel_sampler,) = loader.run(
                "unlabeled",
                ratio=label_ratio,
                runtime=runtime,
            )
            return (label_loader1, label_dataset, label_sampler),  \
                    (test_loader, test_dataset, test_sampler), \
                    (anchor_loader, anchor_dataset, anchor_sampler), \
                    (unlabel_loader, unlabel_dataset, unlabel_sampler), \
                    loader
            
        raise NotImplementedError("fully labeled data not supported in this loader")
        return (label_loader1, label_dataset, label_sampler),  (test_loader, test_dataset, test_sampler), (anchor_loader, anchor_dataset, anchor_sampler)
    elif name=='chest_xray_ssl_acpl':
        diffusion_ds= ChestXRay_nih_ssl(mode='train', 
                                training_platform = training_platform, 
                                labeled = False)
        ## for diffusion part we take all data = labeled + unlabeled
        diffusion_loader = DataLoader(
        diffusion_ds,
        sampler=RandomSampler(diffusion_ds),
        batch_size=train_batches[0],
        num_workers=num_workers,
        drop_last=True)

        ## for acpl - labeled data plus anchor/unlabeled loaders
        label_ratio=2
        runtime=3
        warnings.warn('&&&&&&&&&&&&&&&&&& ACPL DATASETS - REMEMBER WE HAVE LABEL RATIO AND RUNTIME HARDCODED AND MOCKED IN DIFFERENT PLACES self.mock_acpl_args()!!!!!!!!!')
        loader = ChestACPLDataloader(
            batch_size=train_batches[0],
            num_workers=num_workers,
            training_platform=training_platform,
        )
        (test_loader, test_dataset, test_sampler) = loader.run(
            "test",
            ratio=label_ratio,
            runtime=runtime,
        )
        
        (label_loader1, label_dataset, label_sampler,) = loader.run(
            "labeled",
            ratio=label_ratio,
            runtime=runtime,
        )
        
        (anchor_loader, anchor_dataset, anchor_sampler,) = loader.run(
            "anchor",
            ratio=label_ratio,
            runtime=runtime,
        )

        if label_ratio != 100:
            (unlabel_loader, unlabel_dataset, unlabel_sampler,) = loader.run(
                "unlabeled",
                ratio=label_ratio,
                runtime=runtime,
            )
            return diffusion_loader, \
                (label_loader1, label_dataset, label_sampler),  \
                (test_loader, test_dataset, test_sampler), \
                (anchor_loader, anchor_dataset, anchor_sampler), \
                (unlabel_loader, unlabel_dataset, unlabel_sampler), \
                loader
            
        raise NotImplementedError("fully labeled data not supported in this loader")

    elif name=='isic2019_encoder':
        train_ds = ISIC2019(mode='train', training_platform = training_platform, extend_with_test=True, val_split_ratio=0)
        val_ds = ISIC2019(mode='test', training_platform = training_platform)
        test_ds = ISIC2019(mode='test', training_platform = training_platform)
        return train_test_val_dl(
            train_ds, val_ds, test_ds, train_batches, val_batch, num_workers, 
            pin_memory=pin_memory, persistent_workers=persistent_workers)

    # elif name == "cifar10_randaugment":
    #     if num_labeled is not None:
    #         if len(train_batches) != 1:
    #             raise ValueError("Need 1 train batch size - supervised batch size; unsupervised bs = train bs * 7")
    #         args = RandAugmentArgs(num_labeled=num_labeled, num_classes=10, batch_size=train_batches[0])
    #         labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS["cifar10"](args, './data')
    #         return ssl_randaugment_dl(labeled_dataset, unlabeled_dataset, test_dataset, train_batches[0], val_batch, num_workers)
    #     else:
    #         if len(train_batches) != 1:
    #             raise ValueError("Need 1 train batch size - supervised batch size")
    #         args = RandAugmentArgs(num_labeled=num_labeled, num_classes=10, batch_size=train_batches[0])
    #         labeled_dataset, test_dataset = DATASET_GETTERS["cifar10_supervised"](args, './data')
    #         return randaugment_dl(labeled_dataset, test_dataset, train_batches[0], val_batch, num_workers)
    # elif name == "cifar100":
    #     train_ds = AdjustedCIFAR100(train=True)
    #     val_ds = AdjustedCIFAR100(train=False)
    #     num_classes = 100
    #     return non_randaugment_dl(
    #         train_ds, val_ds, num_labeled, train_batches, val_batch, num_classes, num_workers)
    # elif name == "cifar100_randaugment":
    #     if num_labeled is not None:
    #         if len(train_batches) != 1:
    #             raise ValueError("Need 1 train batch size - supervised batch size; unsupervised bs = train bs * 7")
    #         args = RandAugmentArgs(num_labeled=num_labeled, num_classes=100, batch_size=train_batches[0])
    #         labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS["cifar100"](args, './data')
    #         return ssl_randaugment_dl(labeled_dataset, unlabeled_dataset, test_dataset, train_batches[0], val_batch, num_workers)
    #     else:
    #         if len(train_batches) != 1:
    #             raise ValueError("Need 1 train batch size - supervised batch size")
    #         args = RandAugmentArgs(num_labeled=num_labeled, num_classes=10, batch_size=train_batches[0])
    #         labeled_dataset, test_dataset = DATASET_GETTERS["cifar100_supervised"](args, './data')
    #         return randaugment_dl(labeled_dataset, test_dataset, train_batches[0], val_batch, num_workers)
    # elif name == "svhn":
    #     train_ds = AdjustedSVHN(train="train")
    #     val_ds = AdjustedSVHN(train="test")
    #     num_classes = 10
    #     return non_randaugment_dl(
    #         train_ds, val_ds, num_labeled, train_batches, val_batch, num_classes, num_workers)
    # elif name == "svhn_randaugment":
    #     if num_labeled is not None:
    #         if len(train_batches) != 1:
    #             raise ValueError("Need 1 train batch size - supervised batch size; unsupervised bs = train bs * 7")
    #         args = RandAugmentArgs(num_labeled=num_labeled, num_classes=10, batch_size=train_batches[0])
    #         labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS["svhn"](args, './data')
    #         return ssl_randaugment_dl(labeled_dataset, test_dataset, train_batches[0], val_batch, num_workers)
    #     else:
    #         if len(train_batches) != 1:
    #             raise ValueError("Need 1 train batch size - supervised batch size")
    #         args = RandAugmentArgs(num_labeled=num_labeled, num_classes=10, batch_size=train_batches[0])
    #         labeled_dataset, test_dataset = DATASET_GETTERS["svhn_supervised"](args, './data')
    #         return randaugment_dl(labeled_dataset, test_dataset, train_batches[0], val_batch, num_workers)
    # elif name == "mnist":
    #     train_ds = AdjustedMNIST(train=True)
    #     val_ds = AdjustedMNIST(train=False)
    #     num_classes = 10
    #     return non_randaugment_dl(
    #         train_ds, val_ds, num_labeled, train_batches, val_batch, num_classes, num_workers)
    # elif name == "fashion_mnist":
    #     train_ds = AdjustedFashionMNIST(train=True)
    #     val_ds = AdjustedFashionMNIST(train=False)
    #     num_classes = 10
    #     return non_randaugment_dl(
    #         train_ds, val_ds, num_labeled, train_batches, val_batch, num_classes, num_workers)
    else:
        raise NotImplementedError(f"Dataset {name} not implemented")

def train_test_val_dl(train_ds: torch.utils.data.Dataset,
                       val_ds: torch.utils.data.Dataset,
                       test_ds: torch.utils.data.Dataset,
                       train_batches: Tuple[int],
                       val_batch: int,
                       num_workers: int,
                       pin_memory: bool = False,
                       persistent_workers: bool =False,
                       ):
        
        # labels_for_sampler = torch.tensor(train_ds.final_image_df.loc[:, [*train_ds.labels, "no_finding"]].to_numpy(dtype=int))
        # train_dl = torch.utils.data.DataLoader(
        #     train_ds,
        #     batch_size=train_batches[0],
        #     num_workers=num_workers,
        #     drop_last=True,
        #     pin_memory=pin_memory,
        #     persistent_workers = persistent_workers,
        #     sampler = RandomClassSampler(labels=labels_for_sampler)
        # )

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_batches[0],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers = persistent_workers,
    )

    valid_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=val_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers = persistent_workers
    )

    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=val_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers = persistent_workers
    )

    return train_dl, valid_dl, test_dl


def ssl_basic_dl(labeled_dataset, unlabeled_dataset, val_dataset, test_dataset, batch_train, batch_val, num_workers):
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=RandomSampler(labeled_dataset),
        batch_size=batch_train,
        num_workers=num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=RandomSampler(unlabeled_dataset),
        batch_size=batch_train,
        num_workers=num_workers,
        drop_last=True)

    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_val,
        num_workers=num_workers)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_val,
        num_workers=num_workers)
    print("LENGTHS of train lab, train unlab, test", len(labeled_dataset), len(unlabeled_dataset), len(val_dataset), len(test_dataset))
    return (labeled_trainloader, unlabeled_trainloader), val_loader, test_loader


def randaugment_dl(labeled_dataset, test_dataset, batch_train, batch_val, num_workers):
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=RandomSampler(labeled_dataset),
        batch_size=batch_train,
        num_workers=num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_val,
        num_workers=num_workers)
    return labeled_trainloader, test_loader

def supervised_randaugment_dl(labeled_dataset, test_dataset, batch_train, batch_val, num_workers):
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=RandomSampler(labeled_dataset),
        batch_size=batch_train,
        num_workers=num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_val,
        num_workers=num_workers)
    return labeled_trainloader, test_loader


def get_cl_datasets(
        name: str,
        num_labeled: int,
        sup_batch: int,
):
    if name == "cifar10_randaugment":
        if num_labeled is not None:
            args = RandAugmentArgs(
                num_labeled=num_labeled, num_classes=10, batch_size=sup_batch)
            labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS["cifar10"](args, './data', labels_to_tensor=True)
            tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
            sup_tasks_indices = cl_class_split(labeled_dataset.targets, tasks)
            unsup_tasks_indices = cl_class_split(unlabeled_dataset.targets, tasks)
            tasks_datasets = [
                (Subset(labeled_dataset, sup_task_idx), Subset(unlabeled_dataset, unsup_task_idx))
                for sup_task_idx, unsup_task_idx in zip(sup_tasks_indices, unsup_tasks_indices)
            ]
            return tasks_datasets, test_dataset, tasks
        else:
            raise NotImplementedError
    elif name == "cifar100_randaugment":
        if num_labeled is not None:
            args = RandAugmentArgs(
                num_labeled=num_labeled, num_classes=100, batch_size=sup_batch)
            labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS["cifar100"](args, './data', labels_to_tensor=True)
            tasks = [[j for j in range(i*10, (i+1)*10)] for i in range(10)]
            sup_tasks_indices = cl_class_split(labeled_dataset.targets, tasks)
            unsup_tasks_indices = cl_class_split(unlabeled_dataset.targets, tasks)
            tasks_datasets = [
                (Subset(labeled_dataset, sup_task_idx), Subset(unlabeled_dataset, unsup_task_idx))
                for sup_task_idx, unsup_task_idx in zip(sup_tasks_indices, unsup_tasks_indices)
            ]
            return tasks_datasets, test_dataset, tasks
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError(f"Dataset {name} not implemented")
