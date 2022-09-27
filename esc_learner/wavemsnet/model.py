from typing import Union, Tuple

import torch
from torch import nn

from esc_learner.envnet.model import FullyConnectedReLU


class Conv2DBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        **kwargs,
    ) -> None:
        super(Conv2DBatchNorm, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.module(X)


class ResolutionBranch(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], stride: int, pooling_kernel: int
    ):
        super(ResolutionBranch, self).__init__()
        self.conv1 = Conv2DBatchNorm(in_channels, out_channels, kernel_size, stride=stride)
        self.conv2 = Conv2DBatchNorm(out_channels, out_channels, (1, 11), stride=(1, 1))
        self.pool2 = nn.MaxPool1d(pooling_kernel, padding=6)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv2(self.conv1(X))
        return self.pool2(torch.squeeze(X, dim=2))  # self.pool2(X)


class FeatureBlock(nn.Module):
    def __init__(self):
        super(FeatureBlock, self).__init__()
        self.conv1 = Conv2DBatchNorm(1, 64, (3, 3), stride=(1, 1))
        self.pool1 = nn.MaxPool2d((3, 11))
        self.conv2 = Conv2DBatchNorm(64, 128, (3, 3), stride=(1, 1))
        self.pool2 = nn.MaxPool2d((2, 2))
        self.conv3 = Conv2DBatchNorm(128, 256, (3, 3), stride=(1, 1))
        self.pool3 = nn.MaxPool2d((2, 2))
        self.conv4 = Conv2DBatchNorm(256, 256, (3, 3), stride=(1, 1))
        self.pool4 = nn.MaxPool2d((2, 2))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv1(X)
        X = self.pool1(X)
        X = self.conv2(X)
        X = self.pool2(X)
        X = self.conv3(X)
        X = self.pool3(X)
        X = self.conv4(X)
        return self.pool4(X)


class WaveMsNet(nn.Module):
    def __init__(self, n_classes: int):
        super(WaveMsNet, self).__init__()
        self.n_classes = n_classes
        self.branch1 = ResolutionBranch(1, 32, (1, 11), 1, 150)
        self.branch2 = ResolutionBranch(1, 32, (1, 51), 5, 30)
        self.branch3 = ResolutionBranch(1, 32, (1, 101), 10, 15)
        self.block = FeatureBlock()
        self.flatten = nn.Flatten(1, -1)
        self.fc1 = FullyConnectedReLU(256 * 2 * 3, 4096)
        self.fc2 = nn.Linear(4096, n_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.unsqueeze(X, dim=1)
        b1 = self.branch1(X)
        b2 = self.branch2(X)
        b3 = self.branch3(X)
        X = torch.unsqueeze(torch.cat((b1, b2, b3), dim=1), dim=1)
        X = self.block(X)
        return self.fc2(self.fc1(self.flatten(X)))
