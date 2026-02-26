import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from diffusion.scheduler import DiffusionScheduler
from models.unet import SimpleCNN


def compute_rate_distortion(model_path="model.pth", timesteps=1000):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scheduler = DiffusionScheduler(timesteps)
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    x0, _ = next(iter(loader))
    x0 = x0.to(device)

    distortions = []
    rates = []

    cumulative_rate = 0

    with torch.no_grad():

        for t in reversed(range(timesteps)):

            alpha_bar = scheduler.alpha_bar[t].to(device)

            noise = torch.randn_like(x0)
            xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

            t_tensor = torch.tensor([t], device=device)

            pred_noise = model(xt, t_tensor)

            # reconstruction formula (Eq. 15)
            x0_hat = (xt - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)

            # distortion (RMSE)
            rmse = torch.sqrt(torch.mean((x0 - x0_hat) ** 2)).item()
            distortions.append(rmse)

            # approximate rate (noise prediction error)
            mse = torch.mean((noise - pred_noise) ** 2).item()
            cumulative_rate += mse
            rates.append(cumulative_rate)

    distortions = distortions[::-1]
    rates = rates[::-1]

    return distortions, rates


def plot_curves(distortions, rates):

    steps = list(range(len(distortions)))

    plt.figure(figsize=(15,4))

    # Distortion vs steps
    plt.subplot(1,3,1)
    plt.plot(steps, distortions)
    plt.xlabel("Reverse process steps (T - t)")
    plt.ylabel("Distortion (RMSE)")
    plt.title("Distortion vs Steps")

    # Rate vs steps
    plt.subplot(1,3,2)
    plt.plot(steps, rates)
    plt.xlabel("Reverse process steps (T - t)")
    plt.ylabel("Rate (proxy)")
    plt.title("Rate vs Steps")

    # Rate-Distortion
    plt.subplot(1,3,3)
    plt.plot(rates, distortions)
    plt.xlabel("Rate (proxy)")
    plt.ylabel("Distortion (RMSE)")
    plt.title("Rate-Distortion Curve")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    distortions, rates = compute_rate_distortion()
    plot_curves(distortions, rates)