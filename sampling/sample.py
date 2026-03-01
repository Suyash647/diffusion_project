import torch
from torchvision.utils import save_image

from diffusion.scheduler import DiffusionScheduler
from models.unet import UNet


def sample():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scheduler = DiffusionScheduler(timesteps=1000)

    model = UNet(base_channels=128).to(device)
    model.load_state_dict(torch.load("model_ema.pth", map_location=device))
    model.eval()

    x = torch.randn(64, 3, 32, 32).to(device)

    for t in reversed(range(scheduler.timesteps)):

        t_tensor = torch.full(
            (x.shape[0],),
            t,
            device=device,
            dtype=torch.long
        )

        with torch.no_grad():
            eps_theta = model(x, t_tensor)

        alpha = scheduler.alpha[t].to(device)
        alpha_bar = scheduler.alpha_bar[t].to(device)
        beta = scheduler.beta[t].to(device)

        x = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_bar)) * eps_theta
        )

        if t > 0:
            noise = torch.randn_like(x)
            x += torch.sqrt(beta) * noise

    x = (x.clamp(-1, 1) + 1) / 2
    save_image(x, "samples.png", nrow=8)

    print("Saved samples.png")