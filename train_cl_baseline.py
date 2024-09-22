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
from cl_methods.generative_replay import get_replay


if __name__ == "__main__":
    environ["WANDB__SERVICE_WAIT"] = "300"

    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion_config", type=Path, required=True, help="Path to diffusion config file")
    parser.add_argument("--classifier_config", type=Path, required=True, help="Path to classifier config file")
    parser.add_argument("--checkpoint", "-c", type=Path, required=False, help="Path to model checkpoint file")
    parser.add_argument("--old_diffusion", type=Path, required=False, help="Ckpt to old tasks data generator")
    parser.add_argument("--old_classifier", type=Path, required=False, help="Ckpt to old tasks data classifer")
    parser.add_argument("--task", "-t", type=int, required=True, help="Task id")
    parser.add_argument("--learned", "-l", type=int, required=False, help="Learned tasks", nargs="+")
    parser.add_argument("--dir", "-d", type=str, required=False, help="Name for experiments log dir")
    parser.add_argument("--train", type=str, required=True, help="What to train", choices=["diffusion", "classifier"])
    args = parser.parse_args()
    diffusion_config_path = str(args.diffusion_config)
    classifier_config_path = str(args.classifier_config)
    checkpoint_path = str(args.checkpoint) if args.checkpoint is not None else None
    old_diffusion_path = str(args.old_diffusion) if args.old_diffusion is not None else None
    old_classifier_path = str(args.old_classifier) if args.old_classifier is not None else None
    current_task = args.task
    tasks_learned = args.learned if args.learned is not None else []
    to_train = args.train
    custom_dir = args.dir

    diffusion_config = OmegaConf.load(diffusion_config_path)
    classifier_config = OmegaConf.load(classifier_config_path)
    config = diffusion_config if to_train == "diffusion" else classifier_config

    lightning_config = config.pop("lightning", OmegaConf.create())

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["devices"] = -1
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

    cl_config = config.pop("cl")

    reply_buff = get_replay(cl_config.get("reply_type"))(train_bs=tasks_bs, sample_bs=1500, dl_num_workers=16)

    classifier_type = classifier_config.model.get("model_type")
    classifier_params = classifier_config.model.get("params", dict())
    old_classifer = None
    if old_classifier_path is not None:
        classifier_params["ckpt_path"] = old_classifier_path
        old_classifer = get_model_class(classifier_type)(**classifier_params)

    diffusion_type = diffusion_config.model.get("model_type")
    diffusion_params = diffusion_config.model.get("params", dict())
    old_diffusion = None
    if old_diffusion_path is not None:
        diffusion_params["ckpt_path"] = old_diffusion_path
        old_diffusion = get_model_class(diffusion_type)(**diffusion_params)

    model_type = config.model.get("model_type")
    params = config.model.get("params", dict())
    if checkpoint_path is not None:
        config.model.params["ckpt_path"] = checkpoint_path
    model = get_model_class(model_type)(**params)

    model.learning_rate = config.model.base_learning_rate

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = model.__class__.__name__ + "_" + now if custom_dir is None else custom_dir
    logdir = path.join("logs", nowname)
    ckptdir = path.join(logdir, "checkpoints")
    cfgdir = path.join(logdir, "configs")

    trainer_kwargs = dict()
    try:
        per_class = f'{dl_config["train"][0]["cl_split"]["datasets"][0]["ssl_split"]["num_labeled"]} per class'
    except Exception:
        per_class = "all labels"
    tags = [
        dl_config["validation"]["name"],
        per_class,
        config.model.get("model_type"),
        f"task {current_task}",
        f"learned tasks {tasks_learned}",
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

    def get_generator(generator, classifier):
        def generate_samples(batch, labels):
            soft_labels = cl_config.get("use_soft_labels", False)
            ddim = cl_config.get("ddim_steps", False)
            ema = cl_config.get("use_ema", True)
            with torch.no_grad():
                if labels is not None:
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
                pred = classifier(samples)
                pred_labels = pred.argmax(dim=-1)
                if labels is not None:
                    samples = samples[pred_labels == labels]
                    labels = labels[pred_labels == labels] if not soft_labels else pred[pred_labels == labels]
                else:
                    labels = pred_labels if not soft_labels else pred

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
            return samples, labels.cpu()

        return generate_samples

    generate_old_samples = None
    if old_diffusion_path is not None and old_diffusion_path is not None:
        generate_old_samples = get_generator(old_diffusion, old_classifer)

    prev_tasks = []
    for i, (datasets, task) in enumerate(zip(tasks_datasets, tasks)):
        if i == current_task:
            if old_diffusion is not None:
                old_diffusion.to(torch.device("cuda"))
            if old_classifer is not None:
                old_classifer.to(torch.device("cuda"))
            (labeled_ds, unlabeled_ds) = datasets if len(datasets) == 2 else (datasets[0], None)
            train_dls = reply_buff.get_data_for_task(
                sup_ds=labeled_ds,
                unsup_ds=unlabeled_ds,
                prev_tasks=prev_tasks,
                samples_per_task=cl_config["samples_per_class"],
                old_sample_generator=generate_old_samples,
                new_sample_generator=None,  # dummy for class conditioned baseline stuff
                current_task=task,
                filename=nowname,
                saved_samples=cl_config.get("saved_samples", None),
                saved_labels=cl_config.get("saved_labels", None),
            )
            if old_diffusion is not None:
                del old_diffusion
            if old_classifer is not None:
                del old_classifer

            trainer = pl.Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

            weight_reinit = cl_config.get("weight_reinit", "none")
            if weight_reinit == "none":
                pass
            elif weight_reinit == "unused classes":
                torch.nn.init.xavier_uniform_(model.head_out[-1].weight[len(prev_tasks) :])
            elif weight_reinit == "classifier":
                for layer in [model.head_in, model.head_out]:
                    if hasattr(layer, "weight"):
                        torch.nn.init.xavier_uniform_(layer.weight)
            trainer.fit(
                model,
                train_dataloaders=train_dls,
                val_dataloaders=test_dl,
            )
        elif i in tasks_learned:
            prev_tasks.extend(task)
