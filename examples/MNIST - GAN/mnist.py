from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from config import config
from discriminator import DiscriminatorModel
from generator import GeneratorModel
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchscaffold.training.gan_trainer import GANTrainer

GPU_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_data(batch_size: int) -> DataLoader:
    """Return the training data."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])
    trainset = datasets.MNIST(root="~/.pytorch/MNIST_data", train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    return train_loader


def generator_model(path: str) -> GeneratorModel:
    """Return the generator model."""
    generator = GeneratorModel()

    if path is not None and Path(path).exists():
        generator.load_state_dict(torch.load(path))

    generator.to(GPU_DEVICE)
    return generator


def discriminator_model(path: str) -> DiscriminatorModel:
    """Return the discriminator model."""
    discriminator = DiscriminatorModel()

    if path is not None and Path(path).exists():
        discriminator.load_state_dict(torch.load(path))

    discriminator.to(GPU_DEVICE)
    return discriminator


def main():
    # Parse CLI arguments
    args = config()

    # Load models from disk - if they exist
    generator = generator_model(args["generator"])
    discriminator = discriminator_model(args["discriminator"])

    if args["option"] == "train":
        # Define training parameters using CLI arguments
        generator_optimizer = optim.Adam(generator.parameters(), lr=args["lr_generator"])
        discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args["lr_discriminator"])
        criterion = nn.BCELoss()
        train_loader = train_data(args["batch_size"])

        # Train the models
        trainer = GANTrainer(
            generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            criterion,
            train_loader,
            batch_size=args["batch_size"],
            gpu_device=GPU_DEVICE,
        )

        trainer.fit(args["epochs"])

        # Save the models
        torch.save(generator.state_dict(), args["generator"])
        torch.save(discriminator.state_dict(), args["discriminator"])

    if args["option"] == "show":
        # Generate a number
        generator.eval()
        with torch.no_grad():
            _, new_image = generator.random(9, GPU_DEVICE)

        # Show 9x9 grid of images
        fig, ax = plt.subplots(3, 3)
        for i in range(3):
            for j in range(3):
                ax[i, j].axis("off")
                ax[i, j].imshow(new_image[i * 3 + j].squeeze().cpu().detach().numpy(), cmap="gray")

        plt.show()


if __name__ == "__main__":
    main()
