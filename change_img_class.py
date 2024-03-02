from dataclasses import dataclass
from omegaconf import OmegaConf
import argparse
import torch
import pytorch_lightning as pl
from models import get_model_class
from os import path, environ
from pathlib import Path
import datetime
import torchvision as tv
import matplotlib.pyplot as plt
from PIL import Image
from ldm.util import default


if __name__ == "__main__":
    environ["WANDB__SERVICE_WAIT"] = "300"

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--path", "-p", type=Path, required=True, help="path to config file")
    # parser.add_argument("--checkpoint", "-c", type=Path, required=False, help="path to model checkpoint file")
    # args = parser.parse_args()
    # config_path = str(args.path)
    # checkpoint_path = str(args.checkpoint) if args.checkpoint is not None else None

    # config = OmegaConf.load(config_path)
    config = OmegaConf.load("configs/standard_diffusion/semi-supervised/diffmatch_pooling/25_per_class/svhn.yaml")

    # if checkpoint_path is not None:
    #     config.model.params["ckpt_path"] = checkpoint_path
    config.model.params["ckpt_path"] = "./svhn_fid_test.ckpt"

    model = get_model_class(config.model.get("model_type"))(**config.model.get("params", dict()))

    # i = 0
    ds = tv.datasets.CIFAR10("./data", True, tv.transforms.ToTensor())
    # for img, y in ds:
    #     picture = tv.transforms.functional.to_pil_image(img)
    #     picture.save(f"./cifar100/img{i}.png", "PNG")
    #     i += 1
    cuda = torch.device("cuda")
    model.to(cuda)
    model.sampling_method = "conditional_to_x"
    model.sample_grad_scale = 300

    # mean = [0.4914, 0.4822, 0.4465]
    # std = [0.2471, 0.2435, 0.2616]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    batch_size = 53
    num_batches = 1
    classes = 10
    img_list = [f"./cifar10_samples/img{i}.png" for i in range(8, 68) if i not in [62, 54, 53, 46, 45, 37, 31]]
    target_classes = [8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 9, 9, 9, 9, 6, 6, 6, 6, 6, 6, 3, 3, 3, 3, 3]
    t_start = 200

    denormalize = tv.transforms.Compose(
        [
            tv.transforms.Normalize(mean=[0., 0., 0.], std=[1 / s for s in std]),
            tv.transforms.Normalize(mean=[-m for m in mean], std=[1., 1., 1.]),
        ]
    )
    normalize = tv.transforms.Normalize(mean=mean, std=std)

    transform = tv.transforms.ToTensor()
    img_list = [normalize(transform(Image.open(img))).unsqueeze(dim=0) for img in img_list]

    images = torch.concat(img_list).to(model.device)
    noise = default(None, lambda: torch.randn_like(images))
    noised = model.q_sample(images.to(model.device), t=torch.tensor([t_start for _ in range(len(images))], device=model.device).long(), noise=noise)

    model.sample_classes = torch.tensor(target_classes, device=cuda)
    samples = model.sample(len(target_classes), x_start=noised, t_start=t_start)
    samples = samples.cpu()
    samples = denormalize(samples)
    to_show = torch.concat((denormalize(images.cpu()), denormalize(noised.cpu()), samples))
    grid_img = tv.utils.make_grid(to_show, nrow=len(target_classes))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

    unet = model.model.diffusion_model
    representations = unet.just_representations(samples.to(cuda), torch.zeros_like(model.sample_classes), context=None, pooled=False)
    pooled_representations = model.transform_representations(representations)
    pred = model.classifier(pooled_representations)
    print(torch.nn.functional.softmax(pred, dim=-1).max(dim=-1).values, torch.argmax(pred, dim=-1))
