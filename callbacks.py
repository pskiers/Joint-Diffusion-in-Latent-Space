"""
Most of them copied from ldm repo, because python imports
"""

import os
import numpy as np
import time
from PIL import Image
import torchvision
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from omegaconf import OmegaConf
import wandb
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance, adaptive_avg_pool2d
from tqdm import tqdm
from models.adjusted_unet import AdjustedUNet


class FIDScoreLogger(Callback):
    def __init__(
        self,
        device,
        batch_frequency,
        samples_amount,
        metrics_batch_size,
        real_dl,
        real_dl_batch_size,
        dims=2048,
        means_path=None,
        sigma_path=None,
        cond=False,
        latent=True,
        mean=None,
        std=None,
    ) -> None:
        super().__init__()
        self.batch_freq = batch_frequency
        self.batch_real_dl = real_dl_batch_size
        self.real_dl = real_dl
        self.samples_amount = samples_amount
        self.logger_log_fid = {
            pl.loggers.WandbLogger: self._wandb,
        }
        self.device = device
        self.dims = dims
        self.latent = latent
        self.cond = cond
        self.metrics_batch_size = metrics_batch_size
        self.test_mu = None
        self.test_sigma = None
        if means_path is not None and sigma_path is not None:
            self.test_mu = np.load(means_path)
            self.test_sigma = np.load(sigma_path)

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.activation_model = InceptionV3([block_idx]).to(device)
        if (mean is not None) and (std is not None):
            self.denormalize = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / s for s in std]),
                    torchvision.transforms.Normalize(mean=[-m for m in mean], std=[1.0, 1.0, 1.0]),
                ]
            )
        else:
            self.denormalize = None

    def on_train_start(self, trainer, pl_module):
        if self.test_mu is None or self.test_sigma is None:
            real_img_dl = self.real_dl
            img_key = pl_module.first_stage_key if hasattr(pl_module, "first_stage_key") else 0

            self.get_model_statistics(real_img_dl, img_key)

    @torch.no_grad()
    def get_activations(self, dataloader, img_key):
        self.activation_model.eval()

        pred_arr = np.empty((len(dataloader) * self.batch_real_dl, self.dims))

        start_idx = 0

        for batch in tqdm(dataloader, desc="Calculating real FID score"):
            batch = batch[img_key]
            if type(batch) in (list, tuple):
                batch = batch[0]
            else:
                pass
                # batch = batch.permute(0, 3, 1, 2)  # TODO all dataloaders should output in one format (b,c,w,h)
            if self.denormalize is not None:
                batch = self.denormalize(batch)

            # import torchvision as tv
            # import matplotlib.pyplot as plt
            # # samples = self.denormalize(batch)
            # grid = tv.utils.make_grid(batch)
            # plt.imshow(grid.permute(1, 2, 0))
            # plt.show()
            # break
            batch = batch.to(self.device)
            break

            pred = self.activation_model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

            pred_arr[start_idx : start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]

        return pred_arr

    def get_model_statistics(self, dataloader, img_key):
        # self.test_mu = np.zeros((2048,))
        # self.test_sigma = np.zeros((2048, 2048))
        if self.test_mu is None or self.test_sigma is None:
            act = self.get_activations(dataloader, img_key)
            self.test_mu = np.mean(act, axis=0)
            self.test_sigma = np.cov(act, rowvar=False)
        return self.test_mu, self.test_sigma

    def _wandb(self, pl_module, score):
        pl_module.logger.experiment.log({"val/FID": score})

    @torch.no_grad()
    def log_fid(self, pl_module, batch_idx, trainer):
        if pl_module.global_step % self.batch_freq == 0 and self.samples_amount > 0:

            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            real_img_dl = self.real_dl
            img_key = pl_module.first_stage_key if hasattr(pl_module, "first_stage_key") else 0

            m1, s1 = self.get_model_statistics(real_img_dl, img_key)
            # np.save(file="svhn_mean", arr=m1)
            # np.save(file="svhn_s", arr=s1)
            m2, s2 = self.get_samples_statistics(pl_module)

            score = calculate_frechet_distance(m1, s1, m2, s2)

            logger_log_images = self.logger_log_fid.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, score)

            if is_train:
                pl_module.train()

    def get_samples_statistics(self, samples_generator):
        pred_arr = np.empty(
            (
                int(self.samples_amount / self.metrics_batch_size) * self.metrics_batch_size,
                self.dims,
            )
        )

        start_idx = 0

        for _ in tqdm(
            range(int(self.samples_amount / self.metrics_batch_size)),
            desc="Computing FID Score",
        ):
            pred_arr, start_idx = self.get_samples_activations(samples_generator, pred_arr, start_idx)
        mu = np.mean(pred_arr, axis=0)
        sigma = np.cov(pred_arr, rowvar=False)
        return mu, sigma

    @torch.no_grad()
    def get_samples_activations(self, samples_generator, pred_arr, start_idx):
        with samples_generator.ema_scope("Sampling"):
            if self.cond:
                samples_generator.sample_classes = torch.randint(0, 10, [self.metrics_batch_size]).to(self.device)
            if self.latent:
                samples, _ = samples_generator.sample_log(
                    cond=None,
                    batch_size=self.metrics_batch_size,
                    ddim=True,
                    ddim_steps=200,
                    eta=1,
                )
            else:
                samples = samples_generator.sample(batch_size=self.metrics_batch_size, return_intermediates=False)
        if self.latent:
            x_samples = samples_generator.decode_first_stage(samples)
        else:
            x_samples = samples
        if self.denormalize is not None:
            x_samples = self.denormalize(x_samples)
        # import torchvision as tv
        # import matplotlib.pyplot as plt
        # grid = tv.utils.make_grid(x_samples.cpu())
        # plt.imshow(grid.permute(1, 2, 0))
        # plt.show()
        x_samples = x_samples.to(self.device)

        pred = self.activation_model(x_samples)[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        np_pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = np_pred

        start_idx = start_idx + np_pred.shape[0]
        return pred_arr, start_idx

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if pl_module.global_step > 0:
            self.log_fid(pl_module, batch_idx, trainer)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if pl_module.global_step > 0:
            self.log_fid(pl_module, batch_idx, trainer)


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_images,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None,
        log_locally=False,
        mean=None,
        std=None,
    ):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
        }
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_locally = log_locally
        if (mean is not None) and (std is not None):
            self.denormalize = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / s for s in std]),
                    torchvision.transforms.Normalize(mean=[-m for m in mean], std=[1.0, 1.0, 1.0]),
                ]
            )
        else:
            self.denormalize = None

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"images/{split}/{k}"
            pl_module.logger.experiment.log({tag: wandb.Image(grid)})

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (
            self.check_frequency(check_idx)
            and hasattr(pl_module, "log_images")  # batch_idx % self.batch_freq == 0
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    if self.denormalize is not None:
                        images[k] = self.denormalize(images[k])
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            if self.log_locally:
                self.log_local(
                    pl_module.logger.save_dir,
                    split,
                    images,
                    pl_module.global_step,
                    pl_module.current_epoch,
                    batch_idx,
                )

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
            check_idx > 0 or self.log_first_step
        ):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, "calibrate_grad_norm"):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class PerTaskImageLogger(ImageLogger):
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (
            self.check_frequency(check_idx)
            and hasattr(pl_module, "log_images")  # batch_idx % self.batch_freq == 0
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
                samples = images["samples_grad_scale=0"]
                unet: AdjustedUNet = pl_module.model.diffusion_model
                emb = unet.get_timestep_embedding(samples, torch.zeros(len(samples), device=samples.device), None)
                representations = unet.forward_input_blocks(samples, None, emb)
                pooled_representations = pl_module.transform_representations(representations)
                classes = pl_module.classifier(pooled_representations).argmax(dim=-1)

            if hasattr(pl_module, "classes_per_task"):
                for i in range(len(pl_module.old_classes) // pl_module.classes_per_task):
                    task_classes = pl_module.old_classes[
                        i * pl_module.classes_per_task : (i + 1) * pl_module.classes_per_task
                    ]
                    task_mask = torch.any(classes.unsqueeze(-1) == task_classes.to(pl_module.device), dim=-1)
                    if task_mask.sum() == 0:
                        continue
                    task_imgs = samples[task_mask]
                    images[f"task{i}"] = task_imgs
                i += 1
                task_mask = torch.any(classes.unsqueeze(-1) == pl_module.new_classes.to(pl_module.device), dim=-1)
                if task_mask.sum() > 0:
                    task_imgs = samples[task_mask]
                    images[f"task{i}"] = task_imgs
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    if self.denormalize is not None:
                        images[k] = self.denormalize(images[k])
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            if self.log_locally:
                self.log_local(
                    pl_module.logger.save_dir,
                    split,
                    images,
                    pl_module.global_step,
                    pl_module.current_epoch,
                    batch_idx,
                )

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device)
        torch.cuda.synchronize(trainer.strategy.root_device)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device) / 2**20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.strategy.reduce(max_memory)
            epoch_time = trainer.strategy.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config, dl_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.dl_config = dl_config

    def on_exception(self, trainer, pl_module, exception):
        match exception:
            case KeyboardInterrupt():
                if trainer.global_rank == 0:
                    print("Summoning checkpoint...")
                    ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
                    trainer.save_checkpoint(ckpt_path)
                    print("Checkpoint saved")
            case _:
                raise exception

    def setup(self, trainer, pl_module, stage):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if "metrics_over_trainsteps_checkpoint" in self.lightning_config["callbacks"]:
                    os.makedirs(
                        os.path.join(self.ckptdir, "trainstep_checkpoints"),
                        exist_ok=True,
                    )
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)),
            )

            print("Lightning config")
            print(OmegaConf.to_yaml(self.dl_config))
            OmegaConf.save(
                OmegaConf.create({"lightning": self.dl_config}),
                os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)),
            )
            print("Dataloading config")
            print(OmegaConf.to_yaml(self.dl_config))
            OmegaConf.save(
                OmegaConf.create({"dataloading": self.dl_config}),
                os.path.join(self.cfgdir, "{}-dataloading.yaml".format(self.now)),
            )

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """Check if we should save a checkpoint after every train batch"""
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
