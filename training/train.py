import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import torchvision.utils as vutils

from models.unet import UNet
from diffusion.forward import ForwardDiffusion
from diffusion.scheduler import NoiseScheduler
from sampling.sample import sample


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enable fastest convolution algorithms
    torch.backends.cudnn.benchmark = True

    # ------------------------
    # Hyperparameters
    # ------------------------
    num_epochs = 300
    lr = 1e-4
    batch_size = 512
    T_train = 400
    T_sample = 1000

    # ------------------------
    # Model & Optimizer
    # ------------------------
    model = UNet(base_channels=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    # ------------------------
    # Resume Checkpoint
    # ------------------------
    checkpoint_path = "checkpoint_latest.pth"
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    # ------------------------
    # Dataset
    # ------------------------
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
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # ------------------------
    # Diffusion Scheduler
    # ------------------------
    scheduler = NoiseScheduler(T_train)
    forward_diffusion = ForwardDiffusion(scheduler)

    # ------------------------
    # Samples Folder
    # ------------------------
    os.makedirs("samples", exist_ok=True)

    # ------------------------
    # Training Loop
    # ------------------------
    for epoch in range(start_epoch, num_epochs):

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

            # Forward diffusion
            x_noisy, noise = forward_diffusion.add_noise(images, t)

            optimizer.zero_grad()

            # Mixed precision
            with autocast():
                noise_pred = model(x_noisy, t)
                loss = F.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Loss: {avg_loss:.6f}")

        # ------------------------
        # Save Checkpoint
        # ------------------------
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

        print("Checkpoint saved.")

        # ------------------------
        # Generate Samples Every 5 Epochs
        # ------------------------
        if (epoch + 1) % 5 == 0:

            model.eval()

            with torch.no_grad():

                samples = sample(
                    model,
                    T=T_sample,
                    device=device,
                    img_size=28,
                    batch_size=16
                )

                samples = (samples + 1) / 2

                grid = vutils.make_grid(samples, nrow=4)
                vutils.save_image(grid, f"samples/epoch_{epoch+1}.png")

            model.train()
            print("Samples generated.")