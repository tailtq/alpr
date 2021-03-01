import torch
import torch.nn as nn

# Layer type: conv, out_channels, kernel_size, stride
# Layer type: max pooling, stride
ARCHITECTURE = [
    ["conv", 16, 3],
    ["conv", 16, 3],
    ["max_pooling", 2],

    ["conv", 32, 3],
    ["res", 32, 3],
    ["max_pooling", 2],

    ["conv", 64, 3],
    ["res", 64, 3],
    ["res", 64, 3],
    ["max_pooling", 2],

    ["res", 64, 3],
    ["res", 64, 3],
    ["max_pooling", 2],

    ["conv", 128, 3],
    ["res", 128, 3],
    ["res", 128, 3],

    ["detection"]
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding=1, use_activation_fn=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.use_activation_fn = use_activation_fn

    def forward(self, X):
        if self.use_activation_fn:
            return self.relu(self.bn(self.conv(X)))

        return self.bn(self.conv(X))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv1 = CNNBlock(in_channels, out_channels, kernel_size)
        self.conv2 = CNNBlock(out_channels, out_channels, kernel_size, use_activation_fn=False)
        self.relu = nn.ReLU()

    def forward(self, X):
        Y = self.conv1(X)
        Y = self.conv2(Y)
        Y += X

        return self.relu(Y)


class DetectionBlock(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int):
        super().__init__()
        self.conv1 = CNNBlock(in_channels, 2, kernel_size, use_activation_fn=False)
        self.conv2 = CNNBlock(in_channels, 6, kernel_size, use_activation_fn=False)
        self.softmax = nn.Softmax()

    def forward(self, X):
        output1 = self.softmax(self.conv1(X))
        output2 = self.conv2(X)

        return torch.cat([output1, output2], dim=1)


class WPODNet(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels

    def _create_conv_layers(self):
        pass

