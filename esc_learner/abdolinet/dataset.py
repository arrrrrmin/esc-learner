import argparse
from pathlib import Path
from typing import List, TypedDict, Union, Dict, Callable

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def remove_silence(waveform: torch.Tensor, segment_length: int) -> torch.Tensor:
    waveform = waveform[:, waveform.nonzero().min() : waveform.nonzero().max()]
    size = waveform.size(1) + (segment_length - waveform.size(1) % segment_length)
    pad = (int(np.floor((size - waveform.size(1)) / 2)), int(np.ceil((size - waveform.size(1)) / 2)))
    return torch.nn.functional.pad(waveform, pad, mode="constant", value=0)


def segment_waveform(
    waveform: torch.Tensor, segment_length: int, overlap: float, window: Callable = None
) -> torch.Tensor:
    overlap_size = int(np.floor(segment_length * overlap))
    window = window if window else 1.0
    S = [
        waveform[0, j * overlap_size : j * overlap_size + segment_length] * window
        for j in range(int(waveform.size(1) / overlap_size))
    ]
    return torch.stack(S)


class Example(TypedDict):

    audio: torch.Tensor
    label: torch.Tensor


class Folds(Dataset):
    def __init__(self, directory: str, folds: Union[List[int], int], validation: bool, config: argparse.Namespace):
        self.directory = directory
        self.folds = folds if isinstance(folds, list) else [folds]
        self.validation = validation

        self.segment_length = config.segment_length
        self.overlap = config.overlap
        self.window = np.hamming if config.window_fn == "hamming" else None

        self.batch_size = config.batch_size
        self.n_classes = config.n_classes

        self.annotations = pd.read_csv(Path(self.directory) / "meta" / "esc50.csv", delimiter=",")
        self.annotations = self.annotations[self.annotations.fold.map(lambda fold: fold in self.folds)]
        self.data = self.load_folds()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Example:
        return self.data[index]

    def get_target_to_category(self) -> Dict[int, str]:
        return {int(self.annotations.iloc[i, 2]): self.annotations.iloc[i, 3] for i in range(len(self.annotations))}

    def get_category_to_target(self) -> Dict[str, int]:
        return {self.annotations.iloc[i, 3]: int(self.annotations.iloc[i, 2]) for i in range(len(self.annotations))}

    def load_folds(self) -> List[Example]:
        data = []
        for index in range(self.annotations.shape[0]):
            if self.annotations.iloc[index, 1] not in self.folds:
                continue

            path = Path(self.directory) / "audio" / self.annotations.iloc[index, 0]
            waveform, sample_rate = torchaudio.load(path)  # noqa
            waveform = remove_silence(waveform, self.segment_length)
            waveform = segment_waveform(waveform, self.segment_length, self.overlap, self.window)
            label = torch.as_tensor(self.annotations.iloc[index, 2])
            label = torch.nn.functional.one_hot(label, num_classes=self.n_classes).float()
            data.append({"audio": waveform, "label": label})

        return data

    def as_data_loader(self) -> DataLoader:
        return DataLoader(self, batch_size=self.batch_size, shuffle=(not self.validation))
