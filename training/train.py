import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from diffusion.scheduler import DiffusionScheduler
from diffusion.forward import ForwardDiffusion
from models.unet import UNet
from ema import EMA


def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])

    dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

    scheduler = DiffusionScheduler()
    forward = ForwardDiffusion(scheduler)

    model = UNet().to(device)
    ema = EMA(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(100):
        print(f"Epoch {epoch+1}")

        for images, _ in tqdm(loader):

            images = images.to(device)
            t = torch.randint(0, scheduler.timesteps, (images.size(0),), device=device)

            xt, noise = forward.add_noise(images, t)
            pred = model(xt, t)

            loss = torch.mean((noise - pred)**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.update(model)

        print("Loss:", loss.item())

    torch.save(ema.model.state_dict(), "model_cifar10.pth")
    print("Saved EMA model.")