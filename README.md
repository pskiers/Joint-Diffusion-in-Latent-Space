# Joint-Diffusion-in-Latent-Space

## Installing necessary libraries

### Requirements of the installation script
Required components before you start the installation:
* GPU drivers required to run pytorch models on GPU
* Anaconda (or Miniconda) (recommended, but other things like `venv` should work too)

Make sure that Anaconda sees all necessary drivers and libraries, because more often than not it does not and you need to add them manually. For more information see troubleshooting section down bellow.

### Installation
This project was configured in Anaconda. To create environment run:
```
conda env create -f environment.yml 
conda activate jdcl
```
or if you are using other python environment manager make sure you are using python 3.12 or newer and run:
```
pip install -r requirements.txt
```

## Project structure
Project contains 3 main folders:
* configs - some example config files; they are grouped in subfolders accordingly
* dataloading - datasets and dataloader getter for our models
* model - models

## Training a model

### Training diffusion from scratch
It is recommended to have a wandb account created. If you do not then it is recommended to create one. To train a model run:
```
python train_joint_diffusion.py -p {PATH_TO_CONFIG}
```
Logs from the run (metrics, losses, generations) will be available at wandb.ai, model checkpoints will be found in the `./logs` folder.

### Training diffusion from a checkpoint
To resume training from checkpoint run:
```
python train_joint_diffusion.py -p (PATH_TO_CONFIG) -c {PATH_TO CHECKPOINT}
```

### Training autoencoder from scratch
To train autoencoder run:
```
python train_autoencoder.py -p {PATH_TO_CONFIG}
```

### Training autoencoder from a checkpoint
To resume training from checkpoint run:
```
python train_autoencoder.py -p (PATH_TO_CONFIG) -c {PATH_TO CHECKPOINT}
```

## Trouble shooting

### Training script fails
#### Error similar to:
>   ...\
>   File "/home/user/Joint-Diffusion-in-Latent-Space/models/diff_match.py", line 221, in p_losses\
>   &nbsp;&nbsp;&nbsp;  loss, loss_dict = super().p_losses(x_start, cond, t, noise)\
>   File "/home/user/Joint-Diffusion-in-Latent-Space/models/ssl_joint_diffusion.py", line 199, in p_losses\
>   &nbsp;&nbsp;&nbsp;  loss, loss_dict = super().p_losses(x_start, cond, t, noise)\
>   File "/home/user/Joint-Diffusion-in-Latent-Space/models/joint_latent_diffusion_noisy_classifier.py", line 84, in p_losses\
>   &nbsp;&nbsp;&nbsp;  loss, loss_dict = super().p_losses(x_start, cond, t, noise)\
>   File "/home/user/Joint-Diffusion-in-Latent-Space/latent-diffusion/ldm/models/diffusion/ddpm.py", line 1030, in p_losses\
>   &nbsp;&nbsp;&nbsp;  logvar_t = self.logvar[t].to(self.device)\
> RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)\

In that case, what you need to do is insert line:\
`t = t.cpu()`\
in file latent-diffusion/ldm/models/diffusion/ddpm.py, **before line 1030** - just before line `logvar_t = self.logvar[t].to(self.device)`

#### Error ending with:
> packaging.version.InvalidVersion: Invalid version: '0.10.1,<0.11'

In that case run:
```
pip install packaging==21.3
pip install 'torchmetrics<0.8'
```

#### Error:
> AttributeError: module 'torch' has no attribute '_six'

Try running:
```
pip install 'torchvision==0.15.1'
```

#### Error:
```
wandb.errors.UsageError: api_key not configured (no-tty). call wandb.login(key=[your_api_key])
```
Run:
```
wandb login
```
and enter your wandb API key

### Installation script fails
#### Error similar to
```
libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file
```
OR
```
anaconda3/lib/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```
OR
```
ImportError: libmkl_intel_lp64.so.1: cannot open shared object file: No such file or directory
```
It probably means that your Anaconda does not see some libraries. You basically have two options: either install the required libraries directly in your conda environment or add global path to the libraries to conda search paths. Here are some links that you might find useful:
* https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris
* https://stackoverflow.com/questions/48453497/anaconda-libstdc-so-6-version-glibcxx-3-4-20-not-found
* https://stackoverflow.com/questions/67257008/oserror-libmkl-intel-lp64-so-1-cannot-open-shared-object-file-no-such-file-or


