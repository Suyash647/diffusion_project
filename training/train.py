import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import torchvision.utils as vutils

from models.unet import UNet
from diffusion.forward import ForwardDiffusion
from diffusion.scheduler import NoiseScheduler
from ema import EMA
from sampling.sample import sample


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_epochs = 300
    lr = 1e-4
    batch_size = 128
    T_train = 300
    T_sample = 1000

    model = UNet(base_channels=64).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scaler = GradScaler()

    ema = EMA(model)

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    scheduler = NoiseScheduler(T_train)

    forward_diffusion = ForwardDiffusion(scheduler)

    os.makedirs("samples", exist_ok=True)

    for epoch in range(num_epochs):

        print(f"\nEpoch {epoch+1}/{num_epochs}")

        epoch_loss = 0

        for images, _ in tqdm(dataloader):

            images = images.to(device)

            t = torch.randint(
                0,
                T_train,
                (images.size(0),),
                device=device
            ).long()

            x_noisy, noise = forward_diffusion.add_noise(images, t)

            optimizer.zero_grad()

            with autocast():

                noise_pred = model(x_noisy, t)

                loss = F.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            ema.update()

            epoch_loss += loss.item()

        print("Loss:", epoch_loss / len(dataloader))

        if (epoch + 1) % 5 == 0:

            ema.apply_shadow()

            samples = sample(
                model,
                T=T_sample,
                device=device,
                img_size=28,
                batch_size=16
            )

            samples = torch.clamp((samples + 1) / 2, 0, 1)

            grid = vutils.make_grid(samples, nrow=4)

            vutils.save_image(
                grid,
                f"samples/epoch_{epoch+1}.png"
            )

            print("Samples generated.")