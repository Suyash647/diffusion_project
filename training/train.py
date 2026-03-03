import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.unet import UNet
from diffusion.forward import forward_diffusion_sample
from diffusion.scheduler import get_noise_schedule


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------
    # Hyperparameters
    # --------------------
    num_epochs = 300
    lr = 1e-4
    batch_size = 128
    T = 1000  # diffusion steps

    # --------------------
    # Model & Optimizer
    # --------------------
    model = UNet(base_channels=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --------------------
    # Resume Checkpoint
    # --------------------
    checkpoint_path = "checkpoint_latest.pth"
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    # --------------------
    # Dataset
    # --------------------
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
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
        num_workers=2,
        pin_memory=True
    )

    # --------------------
    # Diffusion Scheduler
    # --------------------
    betas = get_noise_schedule(T)

    # --------------------
    # Training Loop
    # --------------------
    for epoch in range(start_epoch, num_epochs):

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_loss = 0

        for images, _ in tqdm(dataloader):

            images = images.to(device)

            t = torch.randint(
                0,
                T,
                (images.size(0),),
                device=device
            ).long()

            x_noisy, noise = forward_diffusion_sample(
                images,
                t,
                betas,
                device
            )

            optimizer.zero_grad()

            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Loss: {avg_loss:.6f}")

        # --------------------
        # Save Checkpoint
        # --------------------
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

        print("Checkpoint saved.\n")