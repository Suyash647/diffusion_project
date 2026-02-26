import torch
import matplotlib.pyplot as plt

from diffusion.scheduler import DiffusionScheduler
from models.unet import SimpleCNN


def sample():

    print("SAMPLE FUNCTION CALLED")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scheduler = DiffusionScheduler(timesteps=1000)

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    # start from pure noise
    x = torch.randn(1, 1, 28, 28).to(device)

    with torch.no_grad():

        for t in reversed(range(scheduler.timesteps)):

            t_tensor = torch.tensor([t], device=device)

            pred_noise = model(x, t_tensor)

            alpha = scheduler.alpha[t]
            alpha_bar = scheduler.alpha_bar[t]
            beta = scheduler.beta[t]

            # reverse mean (Eq. 11)
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise
            )

            # add noise except final step
            if t > 0:
                noise = torch.randn_like(x)
                x += torch.sqrt(beta) * noise

    img = x[0].detach().cpu().squeeze()

    plt.imshow(img, cmap="gray")
    plt.title("Generated Image")
    plt.axis("off")
    plt.show()