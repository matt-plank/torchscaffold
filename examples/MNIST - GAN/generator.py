import torch
import torch.nn as nn

from torchscaffold.layers import ConvTransposeResidualBlock


class GeneratorModel(nn.Module):
    def __init__(self):
        super(GeneratorModel, self).__init__()

        # Input to the network is 7x7 random noise
        self.conv1 = nn.ConvTranspose2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.relu1 = nn.ReLU()

        self.conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.relu2 = nn.ReLU()

        self.block1 = ConvTransposeResidualBlock(in_channels=64, out_channels=64)
        self.block2 = ConvTransposeResidualBlock(in_channels=64, out_channels=64)
        self.block3 = ConvTransposeResidualBlock(in_channels=64, out_channels=64)
        self.block4 = ConvTransposeResidualBlock(in_channels=64, out_channels=64)

        self.conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def random(self, batch_size: int, gpu_device):
        inputs = torch.randn(batch_size, 1, 7, 7, device=gpu_device)

        return inputs, self.forward(inputs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.upsample1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.upsample2(x)
        x = self.relu2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.sigmoid(x)

        return x
