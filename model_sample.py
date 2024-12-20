from omegaconf import OmegaConf
import argparse
import torch
from models import get_model_class, DDIMSamplerGradGuided
from ldm.models.diffusion.ddim import DDIMSampler
from dataloading import get_dataloaders
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
    # config = OmegaConf.load("configs/standard_diffusion/continual_learning/joint_diffusion_pooling/cifar10/cifar10_sigle_task.yaml")
    config = OmegaConf.load("configs/baselines/ddpm/some_points.yaml")

    # if checkpoint_path is not None:
    #     config.model.params["ckpt_path"] = checkpoint_path
    # config.model.params["ckpt_path"] = "logs/MEMORIZATION_CIFAR10v2/checkpoints/last.ckpt"
    config.model.params["ckpt_path"] = "logs/DDPM_2024-12-16T13-17-35/checkpoints/last.ckpt"

    model = get_model_class(config.model.get("model_type"))(**config.model.get("params", dict()))
    # model.register_schedule(
    #     beta_schedule="sqrt_linear",
    #     timesteps=2000,
    #     linear_start=0.001,
    #     linear_end=0.02,
    # )


    cuda = torch.device("cuda")
    model.sampling_method = "unconditional"
    model.sample_grad_scale = 0
    model.sampling_recurence_steps = 1

    # mean = [0.4914, 0.4822, 0.4465]
    # std = [0.2471, 0.2435, 0.2616]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    batch_size = 1
    num_batches = 1
    classes = 20
    ddim_steps = 100

    denormalize = tv.transforms.Compose(
        [
            tv.transforms.Normalize(mean=[0., 0., 0.], std=[1 / s for s in std]),
            tv.transforms.Normalize(mean=[-m for m in mean], std=[1., 1., 1.]),
        ]
    )
    normalize = tv.transforms.Normalize(mean=mean, std=std)

    # torch.manual_seed(42)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False 
    # environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # model.sample_classes = torch.tensor(list(range(20)) * 3 + [0, 1, 2, 3], device=cuda)
    model.to(cuda)
    # model.clip_denoised = False
    # model.sample_classes = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device=cuda)
    x_start = torch.randn((2, 1, 1))
    x_start = torch.stack([x_start] * batch_size)
    # x_start[1] += 0.1
    # max_dist = 0.1
    # x_start += (torch.randn((batch_size, 3, 32, 32))).clamp(-max_dist, max_dist)
    samples = model.sample(batch_size=batch_size, x_start=x_start)
    # ddim_sampler = DDIMSampler(model)
    # shape = (model.channels, model.image_size, model.image_size)
    # samples, _ = ddim_sampler.sample(
    #     S=ddim_steps,
    #     batch_size=batch_size,
    #     shape=shape,
    #     cond=None, 
    #     verbose=False,
    #     x_T=x_start.to(model.device),
    #     repeat_noise=True,
    # )
    print(f"LYAPUNOV STUFF: {sum(model.lyapunov_stuff) / len(model.lyapunov_stuff)}")
    plt.plot([float(l) for l in model.lyapunov_stuff[:-100]])
    plt.show()
    samples = samples.detach().cpu()
    samples = denormalize(samples)
    samples = torch.clamp(samples, 0, 1)

    # unet = model.model.diffusion_model
    # representations = unet.just_representations(samples.to(cuda), torch.zeros(samples.shape[0], device=cuda), context=None, pooled=False)
    # pooled_representations = model.transform_representations(representations)
    # pred = model.classifier(pooled_representations)
    # maxprob, labels = torch.nn.functional.softmax(pred, dim=-1).max(dim=-1).values, torch.argmax(pred, dim=-1)
    # print(maxprob, labels)
    # counts = {i: int((labels == i).sum()) for i in range(20)}
    # for k, v in counts.items():
    #     print(f"{k} -> {v}")
    # # samples = samples[(labels.cpu() == model.sample_classes.cpu())]

    grid_img = tv.utils.make_grid(samples, nrow=16)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
    pass
