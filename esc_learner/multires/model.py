from typing import Union, Tuple

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
        self.conv = Conv2DBatchNorm(1, 64, (3, 3), stride=(1, 1))
        self.pool1 = nn.MaxPool2d((3, 11))  # 32, 40

    def forward(self, X: torch.Tensor):
        return self.pool1(self.conv(X))


class FeatureBlock2(nn.Module):
    def __init__(self):
        super(FeatureBlock2, self).__init__()
        self.conv = Conv2DBatchNorm(42, 128, (3, 3), stride=(1, 1))
        self.pool1 = nn.MaxPool2d((2, 2))  # 16, 20

    def forward(self, X: torch.Tensor):
        return self.pool1(self.conv(X))


class FeatureBlock3(nn.Module):
    def __init__(self):
        super(FeatureBlock3, self).__init__()
        self.conv = Conv2DBatchNorm(128, 256, (3, 3), stride=(1, 1))
        self.pool1 = nn.MaxPool2d((2, 2))  # 4, 5

    def forward(self, X: torch.Tensor):
        return self.pool1(self.conv(X))


class FeatureBlock4(nn.Module):
    def __init__(self):
        super(FeatureBlock4, self).__init__()
        self.conv = Conv2DBatchNorm(256, 256, (3, 3), stride=(1, 1))
        self.pool1 = nn.MaxPool2d((2, 2))  # 8, 10

    def forward(self, X: torch.Tensor):
        return self.pool1(self.conv(X))


class ResolutionBranch(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], stride: int, pooling_kernel: int
    ):
        super(ResolutionBranch, self).__init__()
        self.conv1 = Conv2DBatchNorm(in_channels, out_channels, kernel_size, stride=stride)
        self.conv2 = Conv2DBatchNorm(out_channels, out_channels, (1, 11), stride=(1, 1))
        self.pool2 = nn.MaxPool1d(pooling_kernel, padding=6)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv1(X)
        X = self.conv2(X)
        return self.pool2(torch.squeeze(X, dim=2))


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
        print(X.shape)
        X = self.conv2(X)
        X = self.pool2(X)
        print(X.shape)
        X = self.conv3(X)
        X = self.pool3(X)
        print(X.shape)
        X = self.conv4(X)
        X = self.pool4(X)
        print(X.shape)
        return X


class MultiResCnn(nn.Module):
    def __init__(self, n_classes: int):
        super(MultiResCnn, self).__init__()
        self.n_classes = n_classes
        self.branch1 = ResolutionBranch(1, 32, (1, 11), 1, 150)
        self.branch2 = ResolutionBranch(1, 32, (1, 51), 5, 30)
        self.branch3 = ResolutionBranch(1, 32, (1, 101), 10, 15)
        self.feature = FeatureBlock()
        self.flatten = nn.Flatten(1, -1)
        self.fc1 = FullyConnectedReLU(256 * 2 * 3, 4096)
        self.fc2 = nn.Linear(4096, n_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.unsqueeze(X, dim=1)
        b1 = self.branch1(X)
        # print(b1.shape)
        b2 = self.branch2(X)
        # print(b2.shape)
        b3 = self.branch3(X)
        # print(b3.shape)
        X = torch.unsqueeze(torch.cat((b1, b2, b3), dim=1), dim=1)
        print(X.shape)
        X = self.feature(X)
        # print(X.shape)
        # x = torch.cat((h1, h2, h3, h4), dim=1)
        return self.fc2(self.fc1(self.flatten(X)))


# There are 64, 128, 256, 256 filters in each convolutional layer respectively
# with a size of 3 × 3, and we stride the filter by 1 × 1
if __name__ == "__main__":
    model = MultiResCnn(50)
    size = int(44100 * 1.5 + 100)
    i = torch.rand(4, 1, size)
    o = model(i)
    print(o.shape)
