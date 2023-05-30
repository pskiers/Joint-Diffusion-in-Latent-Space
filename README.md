# Joint-Diffusion-in-Latent-Space

## Creating environment
To create environment run make_env.sh script

## When training does not work
Training script might fail with error similar to:
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
in file latent-diffusion/ldm/models/diffusion/ddpm.py, before line 1030 - just before line `logvar_t = self.logvar[t].to(self.device)`
