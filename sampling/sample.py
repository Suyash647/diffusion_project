import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from diffusion.scheduler import DiffusionScheduler
from diffusion.forward import ForwardDiffusion
from models.unet import UNet


def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    scheduler = DiffusionScheduler(timesteps=1000)
    forward = ForwardDiffusion(scheduler)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    epochs = 100

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        for images, _ in tqdm(loader):

            images = images.to(device)

            t = torch.randint(
                0, scheduler.timesteps,
                (images.shape[0],),
                device=device
            )

            xt, noise = forward.add_noise(images, t)
            pred_noise = model(xt, t)

            loss = torch.mean((noise - pred_noise) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Loss:", loss.item())

    torch.save(model.state_dict(), "model_cifar10.pth")
    print("Model saved.")