import torch


def get_noise_schedule(T, beta_start=1e-4, beta_end=0.02):
    """
    Linear beta schedule used in the original DDPM paper
    """

    betas = torch.linspace(beta_start, beta_end, T)

    return betas