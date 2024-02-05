#!/bin/bash
#SBATCH --job-name=ram_jd_lr105_class102_very_strong_aug_noweight
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgsano4-gpu-a100
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=_job_outputs/%x_job%j.out

module load Miniconda3/4.9.2
nvidia-smi
srun $SCRATCH/ldm/bin/python train_joint_diffusion.py -p configs/latent_diffusion/supervised/non_noisy_classifier/joint_diffusion_pooling/chest_xray_multilabel.yaml --prefix ram_jd_lr105_class101_very_strong_aug_noweight