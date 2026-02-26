import torch


class DiffusionScheduler:
    def __init__(self, timesteps=300):
        self.timesteps = timesteps

        # paper values
        self.beta = torch.linspace(1e-4, 0.02, timesteps)

        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)