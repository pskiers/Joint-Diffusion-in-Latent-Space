#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=03:03:00
#SBATCH --job-name=res
#SBATCH --output=/data/jan_dubinski/Joint-Diffusion-in-Latent-Space/job%j_%x.out

conda activate ldm
python3 /data/jan_dubinski/Joint-Diffusion-in-Latent-Space/res.py