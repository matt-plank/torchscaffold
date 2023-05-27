import torch
import torch.nn as nn

from torchscaffold.layers import ConvResidualBlock


class DiscriminatorModel(nn.Module):
    def __init__(self):
        super(DiscriminatorModel, self).__init__()

        # Residual blocks
        self.block1 = ConvResidualBlock(in_channels=1, out_channels=32)
        self.block2 = ConvResidualBlock(in_channels=32, out_channels=32)
        self.block3 = ConvResidualBlock(in_channels=32, out_channels=32)
        self.block4 = ConvResidualBlock(in_channels=32, out_channels=32)

        # Input shape is 28x28
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()

        # Input shape is 14x14
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()

        # Input shape is 7x7
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(7 * 7 * 16, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pooling1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pooling2(x)
        x = self.relu2(x)

        x = self.flatten(x)
        x = self.layer1(x)

        x = torch.sigmoid(x)
        return x
