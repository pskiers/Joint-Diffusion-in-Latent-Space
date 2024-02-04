#!/bin/bash
#SBATCH --job-name=very_strong_aug_enc_pool8_doubdrop30_lr104_noweight_noft
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgsano4-gpu-a100
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --ntasks-per-node=3
#SBATCH --output=_job_outputs/%x_job%j.out

module load Miniconda3/4.9.2
nvidia-smi
srun $SCRATCH/ldm/bin/python train_classifier_on_ldm.py -p configs/baselines/chest_xray_multilabel_classifier_ldm.yaml --prefix very_strong_aug_enc_pool8_doubdrop_lr104_noweight_noft