#!/bin/bash
#SBATCH --job-name=resnet_only_pool8_doubdrop30_lr104_noweight_noft
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
srun $SCRATCH/ldm/bin/python train_classifier.py -p configs/baselines/chest_xray_multilabel_classifer.yaml --prefix resnet50_only_pool8_doubdrop_lr104_noweight_noft