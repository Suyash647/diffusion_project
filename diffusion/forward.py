import torch


class ForwardDiffusion:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def add_noise(self, x0, t):
        device = x0.device

        alpha_bar = self.scheduler.alpha_bar.to(device)
        alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1)

        noise = torch.randn_like(x0)

        xt = (
            torch.sqrt(alpha_bar_t) * x0 +
            torch.sqrt(1 - alpha_bar_t) * noise
        )

        return xt, noise