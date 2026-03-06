import torch
from tqdm import tqdm


def sample(model, T=1000, device="cuda", img_size=28, batch_size=16, ddim_steps=50):

    model.eval()

    x = torch.randn(batch_size, 1, img_size, img_size).to(device)

    step_size = T // ddim_steps
    timesteps = list(range(0, T, step_size))[::-1]

    for t in tqdm(timesteps):

        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        eps_theta = model(x, t_tensor)

        alpha = 1 - 0.02
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        x = x - eps_theta * 0.1 + noise * 0.01

    return x