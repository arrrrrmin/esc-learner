from collections import OrderedDict

import torch
from typing import Union

import numpy as np
from torch import nn
from torch.nn import functional as f


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


class Transpose(nn.Module):
    def __init__(self, target: int, destination: int) -> None:
        super(Transpose, self).__init__()
        self.target = target
        self.destination = destination

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.transpose(self.target, self.destination)


class FullyConnectedReLU(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super(FullyConnectedReLU, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.module(X)


class EnvNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        """Model object to learn and use EnvNetV1."""
        super(EnvNet, self).__init__()
        self.num_classes = num_classes
        self.feature_conv = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", Conv2DBatchNorm(1, 40, (1, 8))),
                    ("conv2", Conv2DBatchNorm(40, 40, (1, 8))),
                    ("max_pool2", nn.MaxPool2d((1, 160), ceil_mode=False)),
                    ("transpose", Transpose(1, 2)),
                ]
            )
        )
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("conv3", Conv2DBatchNorm(1, 50, (8, 13))),
                    ("max_pool3", nn.MaxPool2d(3, ceil_mode=False)),
                    ("conv4", Conv2DBatchNorm(50, 50, (1, 5))),
                    ("max_pool4", nn.MaxPool2d((1, 3), ceil_mode=False)),
                    ("flatten", nn.Flatten(1, -1)),
                    ("fc5", FullyConnectedReLU(50 * 11 * 14, 4096)),
                    ("fc6", FullyConnectedReLU(4096, 4096)),
                    ("fc7", nn.Linear(4096, self.num_classes)),
                ]
            )
        )

    def freeze_feature_extraction(self):
        for param in self.feature_conv.parameters():
            param.requires_grad = False

    @property
    def t_size(self) -> int:
        return int(np.ceil(self.config.sample_rate * self.config.t)) + 14

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_conv(torch.unsqueeze(X, 1)))

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(X)
        return f.softmax(outputs, dim=-1)

    def extract_features(self, X: torch.Tensor) -> torch.Tensor:
        return self.feature_conv(torch.unsqueeze(X, 1))
