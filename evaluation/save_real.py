import os
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

def save_real_images():

    os.makedirs("real_images", exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for idx, (img, _) in enumerate(loader):
        img = (img.clamp(-1,1) + 1) / 2
        save_image(img, f"real_images/{idx}.png")

    print("Saved real CIFAR-10 test images.")