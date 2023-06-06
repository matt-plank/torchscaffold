from torch import nn

from torchscaffold.layers import ConvTransposeResidualBlock


class UpscalingModel(nn.Module):
    """Model designed to upscale images by a factor of 4."""

    def __init__(self):
        super(UpscalingModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.block1 = ConvTransposeResidualBlock(in_channels=32, out_channels=32)
        self.upscaling1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.block2 = ConvTransposeResidualBlock(in_channels=32, out_channels=32)
        self.block3 = ConvTransposeResidualBlock(in_channels=32, out_channels=32)
        self.upscaling2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.block4 = ConvTransposeResidualBlock(in_channels=32, out_channels=32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.block1(x)
        x = self.upscaling1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.upscaling2(x)
        x = self.block4(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x
