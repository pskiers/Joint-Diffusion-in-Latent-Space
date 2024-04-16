import torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.util import noise_like


class DDIMSamplerGradGuided(DDIMSampler):
    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1,
        noise_dropout=0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1,
        unconditional_conditioning=None,
    ):
        b, *_, device = *x.shape, x.device

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (
            self.model.alphas_cumprod_prev
            if use_original_steps
            else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.model.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.model.ddim_sigmas_for_original_num_steps
            if use_original_steps
            else self.ddim_sigmas
        )
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
        )

        for _ in range(self.model.sampling_recurence_steps - 1):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
                # e_t = self.model.guided_apply_model(x, t)
                if self.model.sampling_method == "conditional_to_x":
                    e_t = self.model.guided_apply_model(x, t)
                elif self.model.sampling_method == "conditional_to_repr":
                    e_t = self.model.guided_repr_apply_model(x, t)
            else:
                raise NotImplementedError()

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(
                    self.model, e_t, x, t, c, **corrector_kwargs
                )
            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.0:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

            # noise back to prev step t
            eps = noise_like(x.shape, x.device)
            alph_t_minus_one = (
                torch.full((b, 1, 1, 1), alphas[index - 1], device=device)
                if index != 0
                else torch.tensor(1, device=x.device, dtype=float)
            )
            alpha_coef = torch.full((b, 1, 1, 1), alphas[index], device=device) / alph_t_minus_one
            x = torch.sqrt(alpha_coef) * x_prev + torch.sqrt(1 - alpha_coef) * eps
        
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            # e_t = self.model.guided_apply_model(x, t)
            if self.model.sampling_method == "unconditional" or self.model.sample_grad_scale == 0:
                unet = self.model.model.diffusion_model
                e_t = unet.just_reconstruction(x, t)
            elif self.model.sampling_method == "conditional_to_x":
                e_t = self.model.guided_apply_model(x, t)
            elif self.model.sampling_method == "conditional_to_repr":
                e_t = self.model.guided_repr_apply_model(x, t)
        else:
            raise NotImplementedError()

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(
                self.model, e_t, x, t, c, **corrector_kwargs
            )
        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
