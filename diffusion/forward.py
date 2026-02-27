import torch

class ForwardDiffusion:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def add_noise(self, x0, t):
        device = x0.device

        # Move schedule tensor to device BEFORE indexing
        alpha_bar = self.scheduler.alpha_bar.to(device)[t]

        sqrt_alpha_bar = torch.sqrt(alpha_bar).view(-1, 1, 1, 1)
        sqrt_one_minus = torch.sqrt(1 - alpha_bar).view(-1, 1, 1, 1)

        noise = torch.randn_like(x0)
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus * noise

        return xt, noise