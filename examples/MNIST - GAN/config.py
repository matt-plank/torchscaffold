import argparse
from pathlib import Path

import yaml


def config():
    # Get config from cli
    parser = argparse.ArgumentParser(description="Train a GAN on MNIST.")
    parser.add_argument("option", choices=["train", "show"], help="Train or test the model.")

    parser.add_argument("--epochs", "-e", type=int, help="Number of epochs to train for.")
    parser.add_argument("--batch-size", "-b", type=int, help="Batch size to use.")
    parser.add_argument("--lr-generator", "--lr-g", type=float, help="Learning rate to use for the generator.")
    parser.add_argument("--lr-discriminator", "--lr-d", type=float, help="Learning rate to use for the discriminator.")

    parser.add_argument("--generator", "-g", type=str, help="Path to generator model.")
    parser.add_argument("--discriminator", "-d", type=str, help="Path to discriminator model.")

    parser.add_argument("--config", "-c", type=str, help="Path to YAML config file.")

    args = parser.parse_args()

    # Get config from YAML
    yaml_config = {}
    if args.config is not None and Path(args.config).exists():
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)

    config = {
        "option": args.option,
        "generator": yaml_config.get("generator", args.generator),
        "discriminator": yaml_config.get("discriminator", args.discriminator),
    }

    if args.option == "train":
        config.update(
            {
                "epochs": int(yaml_config.get("epochs", args.epochs)),
                "batch_size": int(yaml_config.get("batch-size", args.batch_size)),
                "lr_generator": float(yaml_config.get("lr-generator", args.lr_generator)),
                "lr_discriminator": float(yaml_config.get("lr-discriminator", args.lr_discriminator)),
            }
        )

    return config
