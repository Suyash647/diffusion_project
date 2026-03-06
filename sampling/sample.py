import torch
from tqdm import tqdm
from diffusion.scheduler import get_noise_schedule


def sample(model, T=1000, device="cuda", img_size=28, batch_size=16):

    model.eval()

    x = torch.randn(batch_size, 1, img_size, img_size).to(device)

    betas = get_noise_schedule(T).to(device)

    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    for t in tqdm(reversed(range(T))):

        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        eps_theta = model(x, t_tensor)

        alpha = alphas[t]
        alpha_bar_t = alpha_bar[t]

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (
            (1 / torch.sqrt(alpha))
            * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar_t)) * eps_theta)
            + torch.sqrt(betas[t]) * noise
        )

    return x