import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.unet import UNet
from diffusion.scheduler import DiffusionScheduler


device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------
# Load model
# ------------------------------------

model = UNet().to(device)
model.load_state_dict(torch.load("ema_model.pth", map_location=device))
model.eval()

scheduler = DiffusionScheduler(1000).to(device)

# ------------------------------------
# Dataset
# ------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

loader = DataLoader(dataset, batch_size=64, shuffle=False)

# ------------------------------------
# Storage
# ------------------------------------

distortion_list = []
rate_list = []
steps_list = []

# ------------------------------------
# Experiment
# ------------------------------------

for t in tqdm(range(0, scheduler.timesteps, 20)):

    distortion_total = 0
    rate_total = 0
    count = 0

    for x0, _ in loader:

        x0 = x0.to(device)

        noise = torch.randn_like(x0)

        alpha_bar = scheduler.alpha_bar[t]

        xt = (
            torch.sqrt(alpha_bar) * x0
            + torch.sqrt(1 - alpha_bar) * noise
        )

        t_tensor = torch.full((x0.size(0),), t, device=device, dtype=torch.long)

        pred_noise = model(xt, t_tensor)

        # predicted x0
        x0_pred = (
            xt - torch.sqrt(1 - alpha_bar) * pred_noise
        ) / torch.sqrt(alpha_bar)

        # distortion (MSE)
        distortion = torch.mean((x0 - x0_pred) ** 2)

        # rate (approx KL)
        rate = torch.mean(pred_noise ** 2)

        distortion_total += distortion.item()
        rate_total += rate.item()
        count += 1

    distortion_list.append(distortion_total / count)
    rate_list.append(rate_total / count)
    steps_list.append(t)

# ------------------------------------
# Convert to numpy
# ------------------------------------

distortion = np.array(distortion_list)
rate = np.array(rate_list)
steps = np.array(steps_list)

# ------------------------------------
# Plot graphs
# ------------------------------------

plt.figure(figsize=(15,4))

# Distortion vs steps
plt.subplot(1,3,1)
plt.plot(steps, distortion, marker='o')
plt.xlabel("Reverse process steps (T − t)")
plt.ylabel("Distortion (RMSE)")
plt.title("Distortion vs Steps")

# Rate vs steps
plt.subplot(1,3,2)
plt.plot(steps, rate, marker='o')
plt.xlabel("Reverse process steps (T − t)")
plt.ylabel("Rate (bits/dim)")
plt.title("Rate vs Steps")

# Rate–Distortion
plt.subplot(1,3,3)
plt.plot(rate, distortion, marker='o')
plt.xlabel("Rate (bits/dim)")
plt.ylabel("Distortion (RMSE)")
plt.title("Rate–Distortion Curve")

plt.tight_layout()
plt.savefig("rate_distortion_curves.png")
plt.show()