# Joint-Diffusion-in-Latent-Space

## Installing necessary libraries

### A word of caution
It is recommended highly recommended that one uses the provided installation script. Why? Well, the dependencies inherited from the other repositories are a little problematic to say the least. A **python scrip** ending prematurely with a SEGFAULT, scripts not working when the order of imports in one file is changed - these are just a few problems encountered by the author. However, if you still have not been discouraged then I wish you good luck. To be honest you might need it - even if you use the script - because apparently pytorch changing its version might break it (it did once already, even though an older version of pytorch is supposed to be used). If the installing script does not work then do not hesitate to reach out to me and I will try to solve the problem as soon as possible.

### Requirements of the installation script
Required components before you start the installation:
* Anaconda (or Miniconda)
* GPU drivers required to run pytorch models on GPU

Also you should also make sure that Anaconda sees all necessary drivers and libraries, because more often than not it does not and you need to add them manually. For more information see troubleshooting section down bellow.

### Installation
This project was configured in Anaconda. To create environment run:
```
make_env.sh
```
and pray to God it does not fail.

## Project structure
Project contains 3 main folders:
* configs - some example config files; they are grouped in subfolders accordingly
* datasets - datasets and dataloader getter for our models
* model - models

## Training a model

### Training from scratch
It is recommended to have a wandb account created. If you do not then it is recommended to create one. To train a model run:
```
python train_joint_diffusion.py -p {PATH_TO_CONFIG}
```
Logs from the run (metrics, losses, generations) will be available at wandb.ai, model checkpoints will be found in the `./logs` folder.

### Training from a checkpoint
To resume training from checkpoint run:
```
python train_joint_diffusion.py -p (PATH_TO_CONFIG) -c {PATH_TO CHECKPOINT}
```

## Trouble shooting

### Training script fails
Error similar to:
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

### Installation script fails
Error similar to
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


