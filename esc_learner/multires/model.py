from typing import Union

import torch
from torch import nn

from esc_learner.envnet.model import FullyConnectedReLU


class Conv1DBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        **kwargs,
    ) -> None:
        super(Conv1DBatchNorm, self).__init__()
        self.module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.module(X)


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
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.module(X)


class ResolutionBranch1(nn.Module):
    def __init__(self):
        super(ResolutionBranch1, self).__init__()
        self.conv1 = Conv1DBatchNorm(1, 32, 11, stride=1)
        self.conv2 = Conv1DBatchNorm(32, 32, 3, stride=1)
        self.pool2 = nn.MaxPool1d(150)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv1(X)
        X = self.conv2(X)
        return self.pool2(X)


class ResolutionBranch2(nn.Module):
    def __init__(self):
        super(ResolutionBranch2, self).__init__()
        self.conv1 = Conv1DBatchNorm(1, 32, 51, stride=5)
        self.conv2 = Conv1DBatchNorm(32, 32, 3, stride=1)
        self.pool2 = nn.MaxPool1d(30)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv1(X)
        X = self.conv2(X)
        return self.pool2(X)


class ResolutionBranch3(nn.Module):
    def __init__(self):
        super(ResolutionBranch3, self).__init__()
        self.conv1 = Conv1DBatchNorm(1, 32, 101, stride=10)
        self.conv2 = Conv1DBatchNorm(32, 32, 3, stride=1)
        self.pool2 = nn.MaxPool1d(15)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv1(X)
        X = self.conv2(X)
        return self.pool2(X)


class FeatureBlock1(nn.Module):
    def __init__(self):
        super(FeatureBlock1, self).__init__()
        self.conv = Conv2DBatchNorm(1, 64, 3, stride=1, padding="same")
        self.pool1 = nn.MaxPool2d((3, 11))  # 32, 40
        self.pool2 = nn.MaxPool2d((8, 8))  # 4, 5

    def forward(self, X: torch.Tensor):
        return self.pool2(self.pool1(self.conv(X)))


class FeatureBlock2(nn.Module):
    def __init__(self):
        super(FeatureBlock2, self).__init__()
        self.conv = Conv2DBatchNorm(1, 128, 3, stride=1, padding="same")
        self.pool1 = nn.MaxPool2d((6, 22))  # 16, 20
        self.pool2 = nn.MaxPool2d((4, 4))  # 4, 5

    def forward(self, X: torch.Tensor):
        return self.pool2(self.pool1(self.conv(X)))


class FeatureBlock3(nn.Module):
    def __init__(self):
        super(FeatureBlock3, self).__init__()
        self.conv = Conv2DBatchNorm(1, 256, 3, stride=1, padding="same")
        self.pool1 = nn.MaxPool2d((12, 44))  # 8, 10
        self.pool2 = nn.MaxPool2d((2, 2))  # 4, 5

    def forward(self, X: torch.Tensor):
        return self.pool2(self.pool1(self.conv(X)))


class FeatureBlock4(nn.Module):
    def __init__(self):
        super(FeatureBlock4, self).__init__()
        self.conv = Conv2DBatchNorm(1, 256, 3, stride=1, padding="same")
        self.pool1 = nn.MaxPool2d((20, 88))  # 8, 10

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = self.conv(X)
        x = self.pool1(x)
        return x


class MultiResCnn(nn.Module):
    def __init__(self, n_classes: int):
        super(MultiResCnn, self).__init__()
        self.n_classes = n_classes
        self.branch1 = ResolutionBranch1()
        self.branch2 = ResolutionBranch2()
        self.branch3 = ResolutionBranch3()

        self.feature1 = FeatureBlock1()
        self.feature2 = FeatureBlock2()
        self.feature3 = FeatureBlock3()
        self.feature4 = FeatureBlock4()

        self.fc1 = FullyConnectedReLU(704 * 4 * 5, 4096)
        self.fc2 = nn.Linear(4096, n_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        b1, b2, b3 = self.branch1(X), self.branch2(X), self.branch3(X)
        X = torch.unsqueeze(torch.cat((b1, b2, b3), dim=1), dim=1)
        h1 = self.feature1(X)
        h2 = self.feature2(X)
        h3 = self.feature3(X)
        h4 = self.feature4(X)
        x = torch.cat((h1, h2, h3, h4), dim=1)
        return self.fc2(self.fc1(torch.flatten(x, 1, -1)))


# There are 64, 128, 256, 256 filters in each convolutional layer respectively
# with a size of 3 × 3, and we stride the filter by 1 × 1
if __name__ == "__main__":
    model = MultiResCnn(50)
    size = int(44100 * 1.5)
    i = torch.rand(16, 1, size)
    print("input shape", i.shape)
    o = model(i)
    print(o.shape)
