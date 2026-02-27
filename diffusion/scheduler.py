import torch


class DiffusionScheduler:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps

        beta_start = 1e-4
        beta_end = 0.02

        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)