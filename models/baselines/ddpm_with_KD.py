import torch
from ldm.models.diffusion.ddpm import DDPM
from ldm.util import default


class DDPMWithKD(DDPM):
    def __init__(self, old_model, new_model, old_classes, new_classes, *args, kd_weight=1.0, label_key=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_key = label_key
        self.kd_weight = kd_weight
        self.kd_models = {
            "old_model": old_model,
            "new_model": new_model,
        }  # in dict so that pytorch does not see this via parameters()
        self.old_classes = torch.tensor(old_classes, device=self.device)
        self.new_classes = torch.tensor(new_classes, device=self.device)

    def get_input(self, batch, k):
        self.batch_classes = batch[self.label_key]
        return super().get_input(batch, k)

    def p_losses(self, x_start, t, noise=None):
        log_prefix = 'train' if self.training else 'val'

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])
        loss_simple = loss.mean() * self.l_simple_weight
        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss = loss_simple + self.original_elbo_weight * loss_vlb
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})


        if self.kd_weight != 0:
            old_classes_mask = torch.any(self.batch_classes.unsqueeze(-1) == self.old_classes.to(self.device), dim=-1)
            new_classes_mask = torch.any(self.batch_classes.unsqueeze(-1) == self.new_classes.to(self.device), dim=-1)

            # diffusion knowledge distillation
            loss_old = 0
            if old_classes_mask.sum() != 0:
                with torch.no_grad():
                    old_outputs = self.kd_models["old_model"].model(x_noisy[old_classes_mask], t[old_classes_mask])
                loss_old = self.get_loss(model_out[old_classes_mask], old_outputs, mean=True)
            loss_new = 0
            if new_classes_mask.sum() != 0:
                with torch.no_grad():
                    new_outputs = self.kd_models["new_model"].model(x_noisy[new_classes_mask], t[new_classes_mask])
                loss_new = self.get_loss(model_out[new_classes_mask], new_outputs, mean=True)
            loss += self.kd_weight * (loss_new + loss_old)
            loss_dict.update({f'{log_prefix}/loss_diffusion_kl': loss_old + loss_new})

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict
