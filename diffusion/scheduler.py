import torch


class DiffusionScheduler:

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):

        self.timesteps = timesteps

        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # Precompute useful terms
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)