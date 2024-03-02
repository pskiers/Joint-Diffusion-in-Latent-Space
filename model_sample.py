from omegaconf import OmegaConf
import argparse
import torch
from models import get_model_class
from datasets import get_dataloaders
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
    normalize = tv.transforms.Normalize(mean=mean, std=std)


    model.sample_classes = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=cuda)
    samples = model.sample(batch_size)
    samples = samples.cpu()
    samples = denormalize(samples)
    grid_img = tv.utils.make_grid(samples, nrow=1)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

    unet = model.model.diffusion_model
    representations = unet.just_representations(samples.to(cuda), torch.zeros_like(model.sample_classes), context=None, pooled=False)
    pooled_representations = model.transform_representations(representations)
    pred = model.classifier(pooled_representations)
    print(torch.nn.functional.softmax(pred, dim=-1).max(dim=-1).values, torch.argmax(pred, dim=-1))
