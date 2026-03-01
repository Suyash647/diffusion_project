import torch


class ForwardDiffusion:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def add_noise(self, x0, t):
        device = x0.device
        alpha_bar = self.scheduler.alpha_bar.to(device)[t]
        sqrt_ab = torch.sqrt(alpha_bar)[:, None, None, None]
        sqrt_1mab = torch.sqrt(1 - alpha_bar)[:, None, None, None]
        noise = torch.randn_like(x0)
        return sqrt_ab * x0 + sqrt_1mab * noise, noise