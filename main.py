import argparse

from training.train import train
from sampling.sample import sample


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()

    if args.mode == "train":
        train()

    elif args.mode == "sample":
        sample()


if __name__ == "__main__":
    main()