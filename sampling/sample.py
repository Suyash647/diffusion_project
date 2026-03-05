import torch
from torchvision.utils import save_image
from tqdm import tqdm

from diffusion.scheduler import DiffusionScheduler
from models.unet import UNet


def sample():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scheduler = DiffusionScheduler(1000).to(device)

    model = UNet().to(device)
    model.load_state_dict(torch.load("ema_model.pth", map_location=device))
    model.eval()

    x = torch.randn(16, 3, 32, 32).to(device)

    for t in tqdm(reversed(range(scheduler.timesteps))):

        t_tensor = torch.full((16,), t, device=device, dtype=torch.long)

        beta = scheduler.beta[t]
        alpha = scheduler.alpha[t]
        alpha_bar = scheduler.alpha_bar[t]

        pred_noise = model(x, t_tensor)

        x = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise
        )

        if t > 0:
            noise = torch.randn_like(x)
            x += torch.sqrt(beta) * noise

    x = (x.clamp(-1,1) + 1) / 2
    save_image(x, "samples.png", nrow=4)
    print("Saved samples.png")