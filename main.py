print("MAIN STARTED")

from training.train import train
from sampling.sample import sample

mode = "train"   # first run training

if mode == "train":
    print("Starting training...")
    train()

if mode == "sample":
    print("Generating sample...")
    sample()