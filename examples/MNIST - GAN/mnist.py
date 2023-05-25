import argparse

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from discriminator import DiscriminatorModel
from generator import GeneratorModel
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchscaffold.training.gan_trainer import GANTrainer


def train_data(batch_size: int) -> DataLoader:
    """Return the training data."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])
    trainset = datasets.MNIST(root="~/.pytorch/MNIST_data", train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    return train_loader


def cli():
    parser = argparse.ArgumentParser(description="Train a GAN on MNIST.")
    parser.add_argument("option", choices=["train", "show"], help="Train or test the model.")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train for.")
    parser.add_argument("--batch-size", type=int, help="Batch size to use.")
    parser.add_argument("--lr-generator", type=float, help="Learning rate to use for the generator.")
    parser.add_argument("--lr-discriminator", type=float, help="Learning rate to use for the discriminator.")

    parser.add_argument("--number", type=int, help="Number to generate.")

    parser.add_argument("--gpu", type=int, default=None, help="GPU device to use.")

    return parser


def main():
    # Parse CLI arguments
    parser = cli()
    args = parser.parse_args()

    GPU_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.option == "train":
        generator = GeneratorModel()
        generator.to(GPU_DEVICE)
        discriminator = DiscriminatorModel()
        discriminator.to(GPU_DEVICE)

        generator_optimizer = optim.Adam(generator.parameters(), lr=args.lr_generator)
        discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr_discriminator)

        criterion = nn.BCELoss()

        train_loader = train_data(args.batch_size)

        # Training
        trainer = GANTrainer(
            generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            criterion,
            train_loader,
            batch_size=args.batch_size,
            gpu_device=GPU_DEVICE,
        )

        trainer.fit(args.epochs)

        # Show example
        _, new_image = generator.random(9, GPU_DEVICE)

        # Show 9x9 grid of images
        fig, ax = plt.subplots(3, 3)
        for i in range(3):
            for j in range(3):
                ax[i, j].imshow(new_image[i * 3 + j].squeeze().cpu().detach().numpy(), cmap="gray")

        plt.show()


if __name__ == "__main__":
    main()
