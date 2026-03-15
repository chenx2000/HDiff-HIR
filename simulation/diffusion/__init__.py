from .gaussian_diffusion import GaussianDiffusion


def create_diffusion(
        timesteps=4000,
        beta_schedule='linear',
        parameterization='x0',
        clip_denoised=True,
        ddim_sampling_steps='ddim3',
        loss_type='l1',
        v_posterior=0.0,
):
    """
    Factory function to create a GaussianDiffusion instance.

    Args:
        timesteps: Total number of diffusion timesteps.
        beta_schedule: Beta schedule type ('linear' or 'cosine').
        parameterization: Model prediction type ('x0' or 'eps').
        clip_denoised: Whether to clip denoised output to [-1, 1].
        ddim_sampling_steps: DDIM step selection string (e.g., 'ddim3', 'ddim10').
        loss_type: Loss function type ('l1' or 'mse').
        v_posterior: Interpolation factor for posterior variance.

    Returns:
        GaussianDiffusion instance.
    """
    return GaussianDiffusion(
        timesteps=timesteps,
        beta_schedule=beta_schedule,
        parameterization=parameterization,
        clip_denoised=clip_denoised,
        ddim_sampling_steps=ddim_sampling_steps,
        loss_type=loss_type,
        v_posterior=v_posterior,
    )
