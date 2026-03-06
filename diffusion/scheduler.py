import torch
import math


class NoiseScheduler:

    def __init__(self, T):

        self.T = T
        self.betas = self.cosine_beta_schedule(T)

        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def cosine_beta_schedule(self, timesteps, s=0.008):

        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)

        alphas_cumprod = torch.cos(
            ((x / timesteps) + s) / (1 + s) * math.pi * 0.5
        ) ** 2

        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        return torch.clip(betas, 1e-4, 0.999)