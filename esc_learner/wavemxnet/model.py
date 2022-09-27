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


mx_net_configs = {
    "3": [
        Conv1DBatchNorm(1, 256, 80, stride=4),
        nn.MaxPool1d(4),
        Conv1DBatchNorm(256, 256, 3, stride=1, padding="same"),
        nn.MaxPool1d(4),
        nn.AvgPool2d((256, 50)),
        nn.Flatten(1, -1),
    ],
    "5": [
        Conv1DBatchNorm(1, 128, 80, stride=4),
        nn.MaxPool1d(4),
        Conv1DBatchNorm(128, 128, 3, stride=1, padding="same"),
        nn.MaxPool1d(4),
        Conv1DBatchNorm(128, 256, 3, stride=1, padding="same"),
        nn.MaxPool1d(4),
        Conv1DBatchNorm(256, 512, 3, stride=1, padding="same"),
        nn.MaxPool1d(4, ceil_mode=True),
        nn.AvgPool2d((512, 3)),
        nn.Flatten(1, -1),
    ],
    "11": [
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
        nn.AvgPool2d((512, 3)),
        nn.Flatten(1, -1),
    ],
    "18": [
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
        nn.AvgPool2d((512, 3)),
        nn.Flatten(1, -1),
    ],
}


class WaveMxNet(nn.Module):
    def __init__(self, num_classes: int, m: int):
        super(WaveMxNet, self).__init__()
        self.num_classes = num_classes
        if str(m) not in mx_net_configs.keys():
            raise ValueError(f"Config for M{m} not existing, please pass one of {mx_net_configs.keys()}")
        self.m = str(m)
        self.module = nn.Sequential(
            *mx_net_configs[str(m)],
        )
        nn.Softmax(dim=-1),

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for module in self.module:
            # print(module.__class__.__name__)
            X = module(X)
            # print(X.shape)
        return X


if __name__ == "__main__":
    model = WaveMxNet(10, 5)
    x = torch.rand((1, 1, 32000 + 79))  # + 79
    o = model(x)
