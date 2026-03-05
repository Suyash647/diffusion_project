import argparse
from training.train import train
from sampling.sample import sample


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    args = parser.parse_args()

    if args.mode == "train":
        train()

    elif args.mode == "sample":
        sample()