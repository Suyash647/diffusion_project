import torch


def forward_diffusion_sample(x0, t, betas, device):
    """
    Sample x_t from q(x_t | x_0)
    """

    betas = betas.to(device)

    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1)

    noise = torch.randn_like(x0)

    x_t = (
        torch.sqrt(alpha_bar_t) * x0 +
        torch.sqrt(1 - alpha_bar_t) * noise
    )

    return x_t, noise