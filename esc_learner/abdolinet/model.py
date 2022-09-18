from collections import OrderedDict
from typing import Tuple, Union, Literal

import torch
from torch import nn
from torch.nn import functional as f

from esc_learner.envnet.model import FullyConnectedReLU


def majority_vote_aggregator(X: torch.Tensor) -> torch.Tensor:
    return 1 / X.size(0) * X.sum(dim=0)


def sum_rule_aggregator(X: torch.Tensor) -> torch.Tensor:
    return X.sum(dim=0)


class AbdoliNetConvolution(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: int,
        **kwargs,
    ) -> None:
        super(AbdoliNetConvolution, self).__init__()
        self.module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, **kwargs),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.module(X)


class AbdoliNet(nn.Module):
    def __init__(self, num_classes: int, aggregation: Literal["majority_vote", "sum_rule"]) -> None:
        super(AbdoliNet, self).__init__()
        self.num_classes = num_classes
        self.aggregator_fn = majority_vote_aggregator if aggregation == "majority_vote" else sum_rule_aggregator
        self.feature_conv = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", AbdoliNetConvolution(1, 16, (64,), 2)),
                    ("pool1", nn.MaxPool1d((8,), 8)),
                    ("conv2", AbdoliNetConvolution(16, 32, (32,), 2)),
                    ("pool2", nn.MaxPool1d((8,), 8)),
                    ("conv3", AbdoliNetConvolution(32, 64, (16,), 2)),
                    ("conv4", AbdoliNetConvolution(64, 128, (8,), 2)),
                ]
            )
        )
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("flatten", nn.Flatten(1, -1)),
                    ("fc5", FullyConnectedReLU(128 * 8, 128)),
                    ("fc6", FullyConnectedReLU(128, 64)),
                    ("fc7", nn.Linear(64, num_classes)),
                ]
            )
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        y_i = []
        for i in range(X.size(0)):
            y_ij = [
                torch.softmax(self.classifier(self.feature_conv(torch.unsqueeze(X[i][j], dim=0))), dim=-1)
                for j in range(X[i].size(0))
            ]
            y_i.append(self.aggregator_fn(torch.squeeze(torch.stack(y_ij))))
        return torch.stack(y_i, dim=0)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(X)
        return f.softmax(outputs, dim=-1)
