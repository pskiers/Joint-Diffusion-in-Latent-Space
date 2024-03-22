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

# IMPORTANT NOTE
# we moved to pytorch-lightning 1.5.0, torchmetrics 0.7.0, 
# conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# transformers 4.6.0
# packaging 21.3
# rest of the environment needs to be solved according to current conflicts
# if problem with sympy, try to reinstall it first, version 1.12 should work fine