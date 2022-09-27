from typing import Union

import torch
from torch import nn


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


def m3(num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        Conv1DBatchNorm(1, 256, 80, stride=4),
        nn.MaxPool1d(4),
        Conv1DBatchNorm(256, 256, 3, stride=1, padding="same"),
        nn.MaxPool1d(4),
        nn.AvgPool1d(500),
        nn.Flatten(1, -1),
        nn.Linear(256, num_classes),
    )


def m5(num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        Conv1DBatchNorm(1, 128, 80, stride=4),
        nn.MaxPool1d(4),
        Conv1DBatchNorm(128, 128, 3, stride=1, padding="same"),
        nn.MaxPool1d(4),
        Conv1DBatchNorm(128, 256, 3, stride=1, padding="same"),
        nn.MaxPool1d(4),
        Conv1DBatchNorm(256, 512, 3, stride=1, padding="same"),
        nn.MaxPool1d(4, ceil_mode=True),
        nn.AvgPool1d(32),
        nn.Flatten(1, -1),
        nn.Linear(512, num_classes),
    )


def m11(num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        Conv1DBatchNorm(1, 64, 80, stride=4),
        nn.MaxPool1d(4),
        Conv1DBatchNorm(64, 64, 3, stride=1, padding="same"),
        Conv1DBatchNorm(64, 64, 3, stride=1, padding="same"),
        nn.MaxPool1d(4),
        Conv1DBatchNorm(64, 128, 3, stride=1, padding="same"),
        Conv1DBatchNorm(128, 128, 3, stride=1, padding="same"),
        nn.MaxPool1d(4),
        Conv1DBatchNorm(128, 256, 3, stride=1, padding="same"),
        Conv1DBatchNorm(256, 256, 3, stride=1, padding="same"),
        Conv1DBatchNorm(256, 256, 3, stride=1, padding="same"),
        nn.MaxPool1d(4, ceil_mode=True),
        Conv1DBatchNorm(256, 512, 3, stride=1, padding="same"),
        Conv1DBatchNorm(512, 512, 3, stride=1, padding="same"),
        nn.AvgPool1d(32),
        nn.Flatten(1, -1),
        nn.Linear(512, num_classes),
    )


def m18(num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        Conv1DBatchNorm(1, 64, 80, stride=4),
        nn.MaxPool1d(4),
        Conv1DBatchNorm(64, 64, 3, stride=1, padding="same"),
        Conv1DBatchNorm(64, 64, 3, stride=1, padding="same"),
        Conv1DBatchNorm(64, 64, 3, stride=1, padding="same"),
        Conv1DBatchNorm(64, 64, 3, stride=1, padding="same"),
        nn.MaxPool1d(4),
        Conv1DBatchNorm(64, 128, 3, stride=1, padding="same"),
        Conv1DBatchNorm(128, 128, 3, stride=1, padding="same"),
        Conv1DBatchNorm(128, 128, 3, stride=1, padding="same"),
        Conv1DBatchNorm(128, 128, 3, stride=1, padding="same"),
        nn.MaxPool1d(4),
        Conv1DBatchNorm(128, 256, 3, stride=1, padding="same"),
        Conv1DBatchNorm(256, 256, 3, stride=1, padding="same"),
        Conv1DBatchNorm(256, 256, 3, stride=1, padding="same"),
        Conv1DBatchNorm(256, 256, 3, stride=1, padding="same"),
        nn.MaxPool1d(4, ceil_mode=True),
        Conv1DBatchNorm(256, 512, 3, stride=1, padding="same"),
        Conv1DBatchNorm(512, 512, 3, stride=1, padding="same"),
        Conv1DBatchNorm(512, 512, 3, stride=1, padding="same"),
        Conv1DBatchNorm(512, 512, 3, stride=1, padding="same"),
        nn.AvgPool1d(32),
        nn.Flatten(1, -1),
        nn.Linear(512, num_classes),
    )


mx_net_configs = {
    "3": m3,
    "5": m5,
    "11": m11,
    "18": m18,
}


class WaveMxNet(nn.Module):
    def __init__(self, num_classes: int, m: int):
        super(WaveMxNet, self).__init__()
        self.num_classes = num_classes
        if str(m) not in mx_net_configs.keys():
            raise ValueError(f"Config for M{m} not existing, please pass one of {mx_net_configs.keys()}")
        self.m = str(m)
        self.module = nn.Sequential(
            *mx_net_configs[str(m)](num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.module(X)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(X)
        return torch.softmax(outputs, dim=-1)

    @classmethod
    def load_state(cls, num_classes: int, m: int, fp: str) -> "WaveMxNet":
        wavemxnet = cls(num_classes, m)
        wavemxnet.load_state_dict(torch.load(fp))
        # Load a model in eval mode by default
        wavemxnet.eval()
        return wavemxnet


if __name__ == "__main__":
    model = WaveMxNet(10, 18)
    x = torch.rand((1, 1, 32000 + 79))  # + 79
    o = model(x)
