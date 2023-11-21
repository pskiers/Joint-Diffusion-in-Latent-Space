from dataclasses import dataclass
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
    parser.add_argument("--checkpoint", "-c", type=Path, required=False, help="path to model checkpoint file")
    parser.add_argument("--task", "-t", type=int, required=True, help="task id")
    args = parser.parse_args()
    config_path = str(args.path)
    checkpoint_path = str(args.checkpoint) if args.checkpoint is not None else None

    config = OmegaConf.load(config_path)
    # config = OmegaConf.load("configs/standard_diffusion/continual_learning/diffmatch_attention/100_per_class/cifar10.yaml")

    lightning_config = config.pop("lightning", OmegaConf.create())

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["gpus"] = 1
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    dl_config = config.pop("dataloaders")
    reply_buff = GenerativeReplay(
        argparse.Namespace(
            batch_size=dl_config["train_batches"][0],
            sample_batch_size=100,
            num_workers=dl_config["num_workers"],
        )
    )
    tasks_datasets, test_ds, tasks = get_cl_datasets(
        name=dl_config["name"],
        num_labeled=dl_config["num_labeled"],
        sup_batch=dl_config["train_batches"][0],
    )

    test_dl = data.DataLoader(
        test_ds, dl_config["val_batch"], shuffle=False, num_workers=dl_config["num_workers"])

    if checkpoint_path is not None:
        config.model.params["ckpt_path"] = checkpoint_path
    # config.model.params["ckpt_path"] = "./checkpoint"

    model = get_model_class(config.model.get("model_type"))(**config.model.get("params", dict()))

    model.learning_rate = config.model.base_learning_rate

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = model.__class__.__name__ + "_" + now
    logdir = path.join("logs", nowname)
    ckptdir = path.join(logdir, "checkpoints")
    cfgdir = path.join(logdir, "configs")

    trainer_kwargs = dict()
    tags = [
        dl_config["name"],
        "all labels" if dl_config["num_labeled"] is None else f"{dl_config['num_labeled']} per class",
        config.model.get("model_type")
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
            lightning_config=lightning_config
        ),
        CUDACallback()
    ]
    if (img_logger_cfg := callback_cfg.get("img_logger", None)) is not None:
        trainer_kwargs["callbacks"].append(ImageLogger(**img_logger_cfg))

    if (fid_cfg := callback_cfg.get("fid_logger", None)) is not None:
        fid_cfg = dict(fid_cfg)
        fid_cfg["real_dl"] = test_dl
        fid_cfg["device"] = torch.device("cuda")
        trainer_kwargs["callbacks"].append(FIDScoreLogger(**fid_cfg))

    def generate_samples(batch, labels):
        model.gradient_guided_sampling = True
        model.sample_grad_scale = 10
        with torch.no_grad():
            labels = torch.tensor(labels, device=model.device)
            model.sample_classes = labels
            samples = model.sample(batch_size=batch)
            samples = samples.cpu()
            mean = dl_config["mean"]
            std = dl_config["std"]
            denormalize = transforms.Compose(
                [
                    transforms.Normalize(mean=[0., 0., 0.], std=[1 / s for s in std]),
                    transforms.Normalize(mean=[-m for m in mean], std=[1., 1., 1.]),
                ]
            )
            samples = denormalize(samples)
        model.gradient_guided_sampling = False
        model.sample_classes = None
        return samples, labels

    prev_tasks = []
    for i, ((labeled_ds, unlabeled_ds), task) in enumerate(zip(tasks_datasets, tasks)):
        if i == args.task:
            train_dls = reply_buff.get_data_for_task(
                sup_ds=labeled_ds,
                unsup_ds=unlabeled_ds,
                prev_tasks=prev_tasks,
                samples_per_task=2,
                sample_generator=generate_samples,
                filename=dl_config["name"]
            )

            trainer = pl.Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
            trainer.logdir = logdir

            trainer.fit(
                model,
                train_dataloaders=train_dls,
                val_dataloaders=test_dl,
            )
        prev_tasks.extend(task)
