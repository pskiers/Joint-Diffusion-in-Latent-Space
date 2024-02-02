#!/bin/bash

# probably need to run commands one by one 
conda env create -f latent-diffusion/environment.yaml -p /net/tscratch/people/plgjoaxkal/ldm
conda activate ldm
pip install transformers==4.19.2 scann kornia==0.6.4 torchmetrics==0.6.0
pip install git+https://github.com/arogozhnikov/einops.git
pip install wandb
pip install pytorch-fid
pip install matplotlib
pip install --upgrade torch torchvision
pip install git+https://github.com/kornia/kornia
