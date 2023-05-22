import torch
import torch.nn as nn


class GeneratorModel(nn.Module):
    def __init__(self):
        super(GeneratorModel, self).__init__()

        # Input is 64 random numbers (8x8 random image)
        self.conv1 = nn.ConvTranspose2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.upsampling1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.upsampling2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(1)
        self.relu4 = nn.ReLU()

    def random(self, batch_size: int, gpu_device):
        inputs = torch.randn(batch_size, 1, 8, 8).to(gpu_device)

        return inputs, self.forward(inputs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.upsampling1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.upsampling2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = torch.sigmoid(x)
        return x
