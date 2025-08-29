import torch
from torch import nn

class ImageEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Downsample
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)


    def forward(self, x):
        residual = self.downsample(x)
        residual = self.downsample(residual)
        residual = self.downsample(residual)
        residual = self.residual_projection(residual)

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        
        x = x + residual
        x = self.pool(x)
        return x.view(x.size(0), -1)
