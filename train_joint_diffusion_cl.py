from omegaconf import OmegaConf
import argparse
import torch
from torchvision import transforms
import torch.utils.data as data
import pytorch_lightning as pl
from models import get_model_class, DDIMSamplerGradGuided
from dataloading import get_datasets
from os import path, environ
from pathlib import Path
import datetime
from callbacks import (
    ImageLogger,
    CUDACallback,
    SetupCallback,
    FIDScoreLogger,
    CheckpointEveryNSteps,
)
from cl_methods.generative_replay import GenerativeReplay


if __name__ == "__main__":
    environ["WANDB__SERVICE_WAIT"] = "300"

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=Path, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        required=False,
        help="path to model checkpoint file",
    )
    parser.add_argument("--task", "-t", type=int, required=True, help="task id")
    parser.add_argument("--learned", "-l", type=int, required=False, help="Learned tasks", nargs="+")
    parser.add_argument("--new", "-n", type=Path, required=False, help="Ckpt to new task data generator")
    parser.add_argument(
        "--old",
        "-o",
        type=Path,
        required=False,
        help="Ckpt to old tasks data generator",
    )
    parser.add_argument("--tags", type=str, required=False, help="Additional tags", nargs="+")
    args = parser.parse_args()
    config_path = str(args.path)
    # config_path = "configs/standard_diffusion/continual_learning/diffmatch_pooling/25_per_class/cifar10.yaml"
    checkpoint_path = str(args.checkpoint) if args.checkpoint is not None else None
    # checkpoint_path = "pulled_checkpoints/cifar10-t1-s220.ckpt"
    old_generator_path = str(args.old) if args.old is not None else None
    # old_generator_path = None
    new_generator_path = str(args.new) if args.new is not None else None
    # new_generator_path = None
    current_task = args.task
    # current_task = 1
    tasks_learned = args.learned if args.learned is not None else []
    # tasks_learned = [0]
    tags = args.tags

    config = OmegaConf.load(config_path)
    # config = OmegaConf.load("configs/standard_diffusion/continual_learning/diffmatch_pooling/25_per_class/cifar10.yaml")

    lightning_config = config.pop("lightning", OmegaConf.create())

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["gpus"] = 1
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    dl_config_orig = config.pop("dataloaders")
    dl_config = OmegaConf.to_container(dl_config_orig, resolve=True)
    tasks_datasets, tasks_bs, test_ds, test_bs, tasks = get_datasets(dl_config)

    test_dl = data.DataLoader(
        test_ds,
        test_bs,
        shuffle=False,
        num_workers=16,
    )

    reply_buff = GenerativeReplay(train_bs=tasks_bs, sample_bs=250, dl_num_workers=16)

    if checkpoint_path is not None:
        config.model.params["ckpt_path"] = checkpoint_path
    # config.model.params["ckpt_path"] = "logs/DiffMatchFixedPooling_2024-03-11T00-36-09/checkpoints/last.ckpt"

    model = get_model_class(config.model.get("model_type"))(**config.model.get("params", dict()))
    new_generator = None
    if new_generator_path is not None:
        config.model.params["ckpt_path"] = new_generator_path
        new_generator = get_model_class(config.model.get("model_type"))(**config.model.get("params", dict()))
    old_generator = model
    if old_generator_path is not None:
        config.model.params["ckpt_path"] = old_generator_path
        old_generator = get_model_class(config.model.get("model_type"))(**config.model.get("params", dict()))

    model.learning_rate = config.model.base_learning_rate

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = model.__class__.__name__ + "_" + now
    logdir = path.join("logs", nowname)
    ckptdir = path.join(logdir, "checkpoints")
    cfgdir = path.join(logdir, "configs")

    trainer_kwargs = dict()
    try:
        per_class = f'{dl_config["train"][0]["cl_split"]["datasets"][0]["ssl_split"]["num_labeled"]} per class'
    except Exception:
        per_class = "all labels"
    tags.extend(
        [
            dl_config["validation"]["name"],
            per_class,
            config.model.get("model_type"),
            f"task {current_task}",
            f"learned tasks {tasks_learned}",
        ]
    )
    trainer_kwargs["logger"] = pl.loggers.WandbLogger(name=nowname, id=nowname, tags=tags)

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
            "mode": "max",
        }
    }
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 1

    callback_cfg = lightning_config.get("callbacks", OmegaConf.create())
    trainer_kwargs["callbacks"] = [
        pl.callbacks.ModelCheckpoint(**default_modelckpt_cfg["params"]),
        SetupCallback(
            resume=False,
            now=now,
            logdir=logdir,
            ckptdir=ckptdir,
            cfgdir=cfgdir,
            config=config,
            lightning_config=lightning_config,
            dl_config=dl_config_orig,
        ),
        CUDACallback(),
        CheckpointEveryNSteps(10000, prefix="ckpt_"),
    ]
    if (img_logger_cfg := callback_cfg.get("img_logger", None)) is not None:
        trainer_kwargs["callbacks"].append(ImageLogger(**img_logger_cfg))

    if (fid_cfg := callback_cfg.get("fid_logger", None)) is not None:
        fid_cfg = dict(fid_cfg)
        fid_cfg["real_dl"] = test_dl
        fid_cfg["device"] = torch.device("cuda")
        trainer_kwargs["callbacks"].append(FIDScoreLogger(**fid_cfg))

    cl_config = config.pop("cl")

    def get_generator(generator):
        def generate_samples(batch, labels):
            generator.sampling_method = cl_config["sampling_method"]
            generator.sample_grad_scale = cl_config["grad_scale"]
            ddim = cl_config.get("ddim_steps", False)
            ema = cl_config.get("use_ema", True)
            with torch.no_grad():
                labels = torch.tensor(labels, device=generator.device)
                generator.sample_classes = labels
                if not ddim:
                    if ema:
                        with generator.ema_scope():
                            samples = generator.sample(batch_size=batch)
                    else:
                        samples = generator.sample(batch_size=batch)
                else:
                    shape = (
                        generator.channels,
                        generator.image_size,
                        generator.image_size,
                    )
                    if ema:
                        with generator.ema_scope():
                            ddim_sampler = DDIMSamplerGradGuided(generator)
                            samples, _ = ddim_sampler.sample(
                                S=ddim,
                                batch_size=batch,
                                shape=shape,
                                cond=None,
                                verbose=False,
                            )
                    else:
                        ddim_sampler = DDIMSamplerGradGuided(generator)
                        samples, _ = ddim_sampler.sample(
                            S=ddim,
                            batch_size=batch,
                            shape=shape,
                            cond=None,
                            verbose=False,
                        )

                unet = generator.model.diffusion_model
                representations = unet.just_representations(
                    samples,
                    torch.zeros_like(generator.sample_classes),
                    context=None,
                    pooled=False,
                )
                pooled_representations = generator.transform_representations(representations)
                pred = generator.classifier(pooled_representations).argmax(dim=-1)
                ok_class = pred == labels
                samples = samples[ok_class]
                labels = labels[ok_class]

                samples = samples.cpu()
                mean = cl_config["mean"]
                std = cl_config["std"]
                denormalize = transforms.Compose(
                    [
                        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / s for s in std]),
                        transforms.Normalize(mean=[-m for m in mean], std=[1.0, 1.0, 1.0]),
                    ]
                )
                samples = denormalize(samples)
            generator.sampling_method = "unconditional"
            generator.sample_classes = None
            return samples, labels.cpu()

        return generate_samples

    generate_old_samples = get_generator(old_generator)

    new_samples_generate = None
    if new_generator is not None:
        new_samples_generate = get_generator(new_generator)

    prev_tasks = []
    for i, (datasets, task) in enumerate(zip(tasks_datasets, tasks)):
        if i == current_task:
            old_generator.to(torch.device("cuda"))
            if new_generator is not None:
                new_generator.to(torch.device("cuda"))
            (labeled_ds, unlabeled_ds) = datasets if len(datasets) == 2 else (datasets[0], None)
            train_dls = reply_buff.get_data_for_task(
                sup_ds=labeled_ds,
                unsup_ds=unlabeled_ds,
                prev_tasks=prev_tasks,
                samples_per_task=cl_config["samples_per_class"],
                old_sample_generator=generate_old_samples,
                new_sample_generator=(
                    new_samples_generate if unlabeled_ds is not None or new_generator is not None else True
                ),  # dummy for class conditioned baseline stuff
                current_task=task,
                filename=nowname,
                saved_samples=cl_config.get("saved_samples", None),
                saved_labels=cl_config.get("saved_labels", None),
            )
            if new_generator is not None:
                del new_generator
            if old_generator != model:
                del old_generator

            trainer = pl.Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
            trainer.logdir = logdir

            weight_reinit = cl_config.get("weight_reinit", "none")
            if weight_reinit == "none":
                pass
            elif weight_reinit == "unused classes":
                torch.nn.init.xavier_uniform_(model.classifier[-1].weight[len(prev_tasks) :])
            elif weight_reinit == "classifier":
                for layer in model.classifier:
                    if hasattr(layer, "weight"):
                        torch.nn.init.xavier_uniform_(layer.weight)
            trainer.fit(
                model,
                train_dataloaders=train_dls,
                val_dataloaders=test_dl,
            )
        elif i in tasks_learned:
            prev_tasks.extend(task)
