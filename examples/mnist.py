import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchscaffold import layers
from torchscaffold.training import ModelTrainer


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.block1 = layers.ConvResidualBlock(in_channels=32, out_channels=32)
        self.block2 = layers.ConvResidualBlock(in_channels=32, out_channels=32)
        self.block3 = layers.ConvResidualBlock(in_channels=32, out_channels=32)
        self.block4 = layers.ConvResidualBlock(in_channels=32, out_channels=32)

        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(32 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.flatten(x)
        x = self.layer1(x)

        x = torch.softmax(x, dim=1)
        return x


def args_from_cli() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Number of epochs to train for", required=True)
    parser.add_argument("--learning-rate", type=float, help="Learning rate for the optimizer", required=True)
    parser.add_argument("--gpu-device", type=int, help="GPU device to use", default=None)
    args = parser.parse_args()

    return args


def train_test_data() -> tuple[DataLoader, DataLoader]:
    """Return the datasets used for training and testing."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])
    trainset = datasets.MNIST(root="~/.pytorch/MNIST_data", train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.MNIST(root="~/.pytorch/MNIST_data", train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, shuffle=True)

    return train_loader, test_loader


def main():
    """Train a convolutional neural network on the MNIST dataset and plot the training process."""
    args = args_from_cli()

    train_loader, test_loader = train_test_data()

    model = Model()

    if args.gpu_device is not None:
        model.to(args.gpu_device)

    trainer = ModelTrainer(
        model,
        train_loader,
        test_loader,
        optim.Adam(model.parameters(), lr=args.learning_rate),
        nn.CrossEntropyLoss(),
        gpu_device=args.gpu_device,
    )

    results = trainer.fit(args.epochs, args.learning_rate)

    plt.plot(results["training_loss"], label="Loss")
    plt.plot(results["validation_loss"], label="Validation Loss")
    plt.plot(results["validation_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    main()
