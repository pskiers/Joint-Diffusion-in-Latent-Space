from dataclasses import dataclass
from omegaconf import OmegaConf
import argparse
import torch
import pytorch_lightning as pl
from models import get_model_class
from datasets import get_dataloaders
from os import path, environ
from pathlib import Path
import datetime
from callbacks import ImageLogger, CUDACallback, SetupCallback, FIDScoreLogger
import glob

if __name__ == "__main__":
    environ["WANDB__SERVICE_WAIT"] = "300"

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_folder", "-p", type=Path, required=True, help="path to config file")
    parser.add_argument("--enc", "-enc", type=str, required=False, help="path to encoder to easy access the model", 
                        default="/home/jk/Joint-Diffusion-in-Latent-Space/logs/compvis32x32x4_all_randomresizedcrop_Autoencoder_2024-01-17T13-15-27/checkpoints/epoch=000025.ckpt")
    args = parser.parse_args()
    checkpoint_folder = str(args.ckpt_folder)
    config_path = glob.glob(f'{checkpoint_folder}/configs/*project.yaml')[0] 
    config = OmegaConf.load(config_path)
    ckpt_path = glob.glob(f'{checkpoint_folder}/checkpoints/epoch=*.ckpt')[0] 
    config.model.params["ckpt_path"] = ckpt_path
    config.model.params.first_stage_config.params.ckpt_path = args.enc
        

    trainer_config = {"gpus": -1,
                      "accelerator": "ddp"}
    trainer_opt = argparse.Namespace(**trainer_config)
    
    dl_config = {"name": "chest_xray_nih_patches",
                 "train_batches": [None],
                "val_batch": 16,
                "num_workers": 6,
                "pin_memory": True,
                "persistent_workers": True,
                "training_platform": "local_sano"}
    
    test_dl = get_dataloaders(**dl_config)[1]
    model = get_model_class(config.model.get("model_type"))(**config.model.get("params", dict()))
    model.learning_rate = config.model.params.learning_rate
    

    print('[WARNING]this script  Hardcoded for patches')
    trainer = pl.Trainer.from_argparse_args(trainer_opt)

    trainer.test(
        model,
        dataloaders=[test_dl]
    )
