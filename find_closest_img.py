from omegaconf import OmegaConf
import argparse
import torch
from models import get_model_class
from os import environ
from pathlib import Path
import torchvision as tv
import matplotlib.pyplot as plt


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

    denormalize = tv.transforms.Compose(
        [
            tv.transforms.Normalize(mean=[0., 0., 0.], std=[1 / s for s in std]),
            tv.transforms.Normalize(mean=[-m for m in mean], std=[1., 1., 1.]),
        ]
    )

    i = 0
    for _ in range(num_batches):
        # model.sample_classes = torch.randint(1, 2, (batch_size,), device=cuda)
        model.sample_classes = torch.tensor([c // 6 for c in range(batch_size)], device=cuda)
        samples = model.sample(batch_size)
        samples = samples.cpu()
        samples = denormalize(samples)
        for sample in samples:
            picture = tv.transforms.functional.to_pil_image(sample)
            picture.save(f"./cifar10_samples/img{i}.png", "PNG")
            i += 1
    bs = 50000
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=4
    )
    for sample in samples:
        closest_img = None
        min_loss = 100000000000
        for i, (x, y) in enumerate(dl):
            torch.cdist()
            losses = torch.nn.functional.mse_loss(x, sample.expand((len(x), 3, 32, 32)), reduction="none").mean(dim=(1, 2, 3))
            val = losses.min()
            if val < min_loss:
                closest_img = x[losses.argmin()]
                min_loss = val

        to_show = torch.concat((sample.unsqueeze(dim=0), closest_img.unsqueeze(dim=0)))
        grid_img = tv.utils.make_grid(to_show, nrow=1)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
