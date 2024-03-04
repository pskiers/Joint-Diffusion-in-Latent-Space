from omegaconf import OmegaConf
import argparse
import torch
from torchvision import transforms
import torch.utils.data as data
import pytorch_lightning as pl
from models import get_model_class
from datasets import get_cl_datasets
from os import path, environ
from pathlib import Path
import datetime
from callbacks import ImageLogger, CUDACallback, SetupCallback, FIDScoreLogger
from cl_methods.generative_replay import GenerativeReplay


if __name__ == "__main__":
    environ["WANDB__SERVICE_WAIT"] = "300"

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=Path, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint", "-c", type=Path, required=False, help="path to model checkpoint file"
    )
    parser.add_argument("--task", "-t", type=int, required=True, help="task id")
    parser.add_argument(
        "--learned", "-l", type=int, required=False, help="Learned tasks", nargs="+"
    )
    parser.add_argument(
        "--new", "-n", type=Path, required=False, help="Ckpt to new task data generator"
    )
    parser.add_argument(
        "--old", "-o", type=Path, required=False, help="Ckpt to old tasks data generator"
    )
    args = parser.parse_args()
    config_path = str(args.path)
    # config_path = "configs/baselines/class_conditioned_ddpm/cifar10.yaml"
    checkpoint_path = str(args.checkpoint) if args.checkpoint is not None else None
    # checkpoint_path = None
    old_generator_path = str(args.old) if args.old is not None else None
    # old_generator_path = None
    new_generator_path = str(args.new) if args.new is not None else None
    # new_generator_path = None
    current_task = args.task
    # current_task = 2
    tasks_learned = args.learned if args.learned is not None else []
    # tasks_learned = []

    config = OmegaConf.load(config_path)
    # config = OmegaConf.load("configs/standard_diffusion/continual_learning/diffmatch_pooling/25_per_class/cifar10.yaml")

    lightning_config = config.pop("lightning", OmegaConf.create())

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["gpus"] = 1
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    dl_config = config.pop("dataloaders")
    reply_buff = GenerativeReplay(
        argparse.Namespace(
            batch_size=dl_config["train_batches"][0],
            sample_batch_size=500,
            num_workers=dl_config["num_workers"],
        )
    )
    tasks_datasets, test_ds, tasks = get_cl_datasets(
        name=dl_config["name"],
        num_labeled=dl_config["num_labeled"],
        sup_batch=dl_config["train_batches"][0],
    )

    test_dl = data.DataLoader(
        test_ds, dl_config["val_batch"], shuffle=False, num_workers=dl_config["num_workers"]
    )

    if checkpoint_path is not None:
        config.model.params["ckpt_path"] = checkpoint_path
    # config.model.params["ckpt_path"] = "./cl_cifar10.ckpt"

    model = get_model_class(config.model.get("model_type"))(**config.model.get("params", dict()))
    new_generator = None
    if new_generator_path is not None:
        config.model.params["ckpt_path"] = new_generator_path
        new_generator = get_model_class(config.model.get("model_type"))(
            **config.model.get("params", dict())
        )
    old_generator = model
    if old_generator_path is not None:
        config.model.params["ckpt_path"] = old_generator_path
        old_generator = get_model_class(config.model.get("model_type"))(
            **config.model.get("params", dict())
        )

    model.learning_rate = config.model.base_learning_rate

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = model.__class__.__name__ + "_" + now
    logdir = path.join("logs", nowname)
    ckptdir = path.join(logdir, "checkpoints")
    cfgdir = path.join(logdir, "configs")

    trainer_kwargs = dict()
    tags = [
        dl_config["name"],
        (
            "all labels"
            if dl_config["num_labeled"] is None
            else f"{dl_config['num_labeled']} per class"
        ),
        config.model.get("model_type"),
    ]
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
            "every_n_train_steps": 10000,
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
        ),
        CUDACallback(),
    ]
    if (img_logger_cfg := callback_cfg.get("img_logger", None)) is not None:
        trainer_kwargs["callbacks"].append(ImageLogger(**img_logger_cfg))

    if (fid_cfg := callback_cfg.get("fid_logger", None)) is not None:
        fid_cfg = dict(fid_cfg)
        fid_cfg["real_dl"] = test_dl
        fid_cfg["device"] = torch.device("cuda")
        trainer_kwargs["callbacks"].append(FIDScoreLogger(**fid_cfg))

    cl_config = config.pop("cl")

    def generate_old_samples(batch, labels):
        old_generator.sampling_method = cl_config["sampling_method"]
        old_generator.sample_grad_scale = cl_config["grad_scale"]
        with torch.no_grad():
            labels = torch.tensor(labels, device=old_generator.device)
            old_generator.sample_classes = labels
            samples = old_generator.sample(batch_size=batch)
            samples = samples.cpu()
            mean = dl_config["mean"]
            std = dl_config["std"]
            denormalize = transforms.Compose(
                [
                    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / s for s in std]),
                    transforms.Normalize(mean=[-m for m in mean], std=[1.0, 1.0, 1.0]),
                ]
            )
            samples = denormalize(samples)
        old_generator.sampling_method = "unconditional"
        old_generator.sample_classes = None
        return samples, labels.cpu()

    new_samples_generate = None
    if new_generator is not None:

        def generate_new_samples(batch, labels):
            new_generator.sampling_method = cl_config["sampling_method"]
            new_generator.sample_grad_scale = cl_config["grad_scale"]
            with torch.no_grad():
                labels = torch.tensor(labels, device=new_generator.device)
                new_generator.sample_classes = labels
                samples = new_generator.sample(batch_size=batch)
                samples = samples.cpu()
                mean = dl_config["mean"]
                std = dl_config["std"]
                denormalize = transforms.Compose(
                    [
                        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / s for s in std]),
                        transforms.Normalize(mean=[-m for m in mean], std=[1.0, 1.0, 1.0]),
                    ]
                )
                samples = denormalize(samples)
            new_generator.sampling_method = "unconditional"
            new_generator.sample_classes = None
            return samples, labels.cpu()

        new_samples_generate = generate_new_samples

    prev_tasks = []
    for i, (datasets, task) in enumerate(zip(tasks_datasets, tasks)):
        if i == current_task:
            old_generator.to(torch.device("cuda"))
            if new_generator is not None:
                new_generator.to(torch.device("cuda"))
            (labeled_ds, unlabeled_ds) = datasets if len(datasets) == 2 else (datasets, None)
            train_dls = reply_buff.get_data_for_task(
                sup_ds=labeled_ds,
                unsup_ds=unlabeled_ds,
                prev_tasks=prev_tasks,
                samples_per_task=cl_config["samples_per_class"],
                old_sample_generator=generate_old_samples,
                new_sample_generator=(
                    new_samples_generate if unlabeled_ds is not None else True
                ),  # dummy for class conditioned baseline stuff
                current_task=task,
                filename=nowname,
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
