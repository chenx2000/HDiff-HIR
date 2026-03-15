"""
Gaussian Diffusion model for training (DDPM) and sampling (DDIM).

This module encapsulates all diffusion process computations:
    - Forward process: q(x_t | x_0) for adding noise during training
    - Reverse process: p(x_{t-1} | x_t) for DDPM / DDIM sampling
    - Training loss computation

Inspired by: https://github.com/openai/improved-diffusion
"""

import torch
import torch.nn.functional as F
from inspect import isfunction

from .schedules import linear_beta_schedule, cosine_beta_schedule, space_timesteps


def _extract_into_tensor(a, t, x_shape):
    """Extract values from tensor `a` at indices `t`, broadcast to `x_shape`."""
    b, *_ = t.shape
    a = a.cuda()
    t = t.cuda()
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def _exists(x):
    return x is not None


def _default(val, d):
    if _exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion:
    """
    Gaussian Diffusion process for training (DDPM) and sampling (DDIM).

    Training uses full DDPM forward process (q_sample) with all timesteps.
    Sampling uses DDIM reverse process with a configurable number of steps.

    Args:
        timesteps: Total number of diffusion timesteps. Default: 4000.
        beta_schedule: Beta schedule type ('linear' or 'cosine'). Default: 'linear'.
        parameterization: Model prediction type ('x0' or 'eps'). Default: 'x0'.
        clip_denoised: Whether to clip denoised output to [-1, 1]. Default: True.
        ddim_sampling_steps: DDIM step selection string (e.g., 'ddim3'). Default: 'ddim3'.
        loss_type: Loss function type ('l1' or 'mse'). Default: 'l1'.
        v_posterior: Interpolation factor for posterior variance (0=small, 1=large). Default: 0.0.
    """

    def __init__(
            self,
            timesteps=4000,
            beta_schedule='linear',
            parameterization='x0',
            clip_denoised=True,
            ddim_sampling_steps='ddim3',
            loss_type='l1',
            v_posterior=0.0,
    ):
        self.timesteps = timesteps
        self.parameterization = parameterization
        self.clip_denoised = clip_denoised
        self.loss_type = loss_type

        # ===========================================================
        # DDPM coefficients (full timesteps, used for training)
        # ===========================================================
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = betas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # q(x_t | x_0) coefficients
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

        # q(x_{t-1} | x_t, x_0) posterior coefficients
        posterior_variance = (
            (1 - v_posterior) * betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            + v_posterior * betas
        )
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            torch.maximum(posterior_variance, torch.tensor(1e-20))
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # ===========================================================
        # DDIM coefficients (sub-sampled timesteps, used for sampling)
        # ===========================================================
        use_timesteps = set(space_timesteps(timesteps, ddim_sampling_steps))
        self.timestep_map = []
        last_alpha_cumprod = 1.0
        new_betas = []

        for i, ac in enumerate(alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - ac / last_alpha_cumprod)
                last_alpha_cumprod = ac
                self.timestep_map.append(i)

        betas_ddim = torch.tensor(new_betas)
        alphas_ddim = 1.0 - betas_ddim
        alphas_cumprod_ddim = torch.cumprod(alphas_ddim, dim=0)
        alphas_cumprod_prev_ddim = F.pad(alphas_cumprod_ddim[:-1], (1, 0), value=1.0)

        self.ddim_alphas_cumprod = alphas_cumprod_ddim
        self.ddim_alphas_cumprod_prev = alphas_cumprod_prev_ddim
        self.ddim_sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod_ddim)
        self.ddim_sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod_ddim - 1)

        # DDIM posterior coefficients
        ddim_posterior_variance = (
            (1 - v_posterior) * betas_ddim * (1.0 - alphas_cumprod_prev_ddim) / (1.0 - alphas_cumprod_ddim)
            + v_posterior * betas_ddim
        )
        self.ddim_posterior_variance = ddim_posterior_variance
        self.ddim_posterior_log_variance_clipped = torch.log(
            torch.maximum(ddim_posterior_variance, torch.tensor(1e-20))
        )
        self.ddim_posterior_mean_coef1 = (
            betas_ddim * torch.sqrt(alphas_cumprod_prev_ddim) / (1.0 - alphas_cumprod_ddim)
        )
        self.ddim_posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev_ddim) * torch.sqrt(alphas_ddim) / (1.0 - alphas_cumprod_ddim)
        )

    # ===============================================================
    # Training: DDPM forward process
    # ===============================================================

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: sample x_t from q(x_t | x_0).

        Args:
            x_start: Clean input tensor x_0 of shape (B, C, H, W).
            t: Timestep tensor of shape (B,).
            noise: Optional pre-generated noise. Default: random Gaussian.

        Returns:
            Noisy tensor x_t of shape (B, C, H, W).
        """
        noise = _default(noise, lambda: torch.randn_like(x_start))
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, model, x_start, t, condition, noise=None, mask=None):
        """
        Compute the training loss for the diffusion model.

        Args:
            model: Denoising network (UNet).
            x_start: Clean input tensor x_0 of shape (B, C, H, W).
            t: Timestep tensor of shape (B,).
            condition: Conditioning input (shifted measurement) of shape (B, C, H, W).
            noise: Optional pre-generated noise.
            mask: Physical mask tensor of shape (B, C, H, W).

        Returns:
            Scalar loss value.
        """
        noise = _default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = model(x_noisy, t, condition, mask=mask)

        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(
                f"Parameterization {self.parameterization} not yet supported"
            )

        if self.loss_type == 'l1':
            loss = F.l1_loss(model_out, target)
        elif self.loss_type == 'mse':
            loss = F.mse_loss(model_out, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    # ===============================================================
    # Sampling: DDIM reverse process
    # ===============================================================

    def _predict_start_from_noise(self, x_t, t, noise):
        """Compute x_0 prediction from noise prediction (eps parameterization)."""
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """Compute noise (eps) from x_0 prediction (x0 parameterization)."""
        return (
            _extract_into_tensor(self.ddim_sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.ddim_sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _q_posterior_ddim(self, x_start, x_t, t):
        """Compute posterior q(x_{t-1} | x_t, x_0) using DDIM coefficients."""
        posterior_mean = (
            _extract_into_tensor(self.ddim_posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.ddim_posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_var = _extract_into_tensor(self.ddim_posterior_variance, t, x_t.shape)
        posterior_log_var = _extract_into_tensor(
            self.ddim_posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_var, posterior_log_var

    def _p_mean_variance_ddim(self, model, x, t, condition, mask=None):
        """
        Compute the model's predicted mean and variance for DDIM sampling.

        Maps sub-sampled timestep indices to original timestep values for the model.
        """
        t = t.cuda()
        t_input = torch.tensor([self.timestep_map[i] for i in t]).cuda()
        model_out = model(x, t_input, condition, mask=mask)

        if self.parameterization == "eps":
            x_recon = self._predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_var, posterior_log_var = self._q_posterior_ddim(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_var, posterior_log_var, model_out

    def _ddim_sample(self, model, x, t, condition, mask=None, eta=0.0):
        """
        Sample x_{t-1} from the model using DDIM.

        Args:
            model: Denoising network.
            x: Current noisy tensor x_t.
            t: Current timestep index tensor.
            condition: Conditioning input.
            mask: Physical mask tensor.
            eta: DDIM stochasticity parameter (0 = deterministic).

        Returns:
            Dict with 'sample' (x_{t-1}) and 'pred_xstart' (predicted x_0).
        """
        _, _, _, model_out = self._p_mean_variance_ddim(
            model, x=x, t=t, condition=condition, mask=mask
        )

        # Re-derive eps from x_0 prediction
        eps = self._predict_eps_from_xstart(x, t, model_out)
        alpha_bar = _extract_into_tensor(self.ddim_alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.ddim_alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # DDIM Equation 12
        noise = torch.randn_like(x).cuda()
        mean_pred = (
            model_out * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": model_out}

    def _ddim_sample_loop_progressive(
            self, model, shape, condition, mask=None,
            noise=None, eta=0.0, progress=True,
    ):
        """
        DDIM sampling loop that yields intermediate results at each step.

        Args:
            model: Denoising network.
            shape: Output tensor shape (B, C, H, W).
            condition: Conditioning input.
            mask: Physical mask tensor.
            noise: Optional initial noise tensor.
            eta: DDIM stochasticity parameter.
            progress: Whether to show progress bar.

        Yields:
            Dict with 'sample' and 'pred_xstart' at each DDIM step.
        """
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape).cuda()
        indices = list(range(len(self.timestep_map)))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0]).cuda()
            with torch.no_grad():
                out = self._ddim_sample(
                    model, x=img, t=t, condition=condition,
                    mask=mask, eta=eta,
                )
                yield out
                img = out["sample"]

    def ddim_sample_loop(
            self, model, shape, condition, mask=None,
            noise=None, eta=0.0, progress=True,
    ):
        """
        Generate samples from the model using DDIM.

        Args:
            model: Denoising network.
            shape: Output tensor shape (B, C, H, W).
            condition: Conditioning input.
            mask: Physical mask tensor.
            noise: Optional initial noise tensor.
            eta: DDIM stochasticity parameter (0 = deterministic).
            progress: Whether to show progress bar.

        Returns:
            Final denoised sample tensor of shape (B, C, H, W).
        """
        final = None
        for sample in self._ddim_sample_loop_progressive(
                model, shape, condition,
                mask=mask, noise=noise, eta=eta, progress=progress,
        ):
            final = sample
        return final["sample"]
