import argparse
from pathlib import Path
from typing import List, TypedDict, Union, Dict

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


annotation_indecies = {"esc50": (1, 2, 3), "urbansound8k": (5, 6, 7)}


def remove_silence(waveform: torch.Tensor, pad_width: int) -> torch.Tensor:
    waveform = waveform[:, waveform.nonzero().min() : waveform.nonzero().max() + 1]
    size = pad_width - waveform.size(1)
    pad = (int(np.floor(size / 2)), int(np.ceil(size / 2)))
    return torch.nn.functional.pad(waveform, pad, mode="constant", value=0)


class Example(TypedDict):

    audio: torch.Tensor
    label: torch.Tensor


class Folds(Dataset):
    def __init__(self, directory: str, folds: Union[List[int], int], validation: bool, config: argparse.Namespace):
        self.directory = directory
        self.folds = folds if isinstance(folds, list) else folds
        self.validation = validation

        self.dataset = config.dataset
        self.max_length = config.max_length

        self.batch_size = config.batch_size
        self.n_classes = config.n_classes

        self.dataset_name = config.dataset
        self.fold_idx, self.target_idx, self.classname_idx = annotation_indecies[config.dataset]
        self.meta_file_name = "esc50.csv" if config.dataset == "esc" else "urbansound8k.csv"
        self.annotations = pd.read_csv(Path(self.directory) / "meta" / self.meta_file_name, delimiter=",")
        self.annotations = self.annotations[self.annotations.fold.map(lambda fold: fold in self.folds)]

        self.data = self.load_folds()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Example:
        return self.data[index]

    def get_target_to_category(self) -> Dict[int, str]:
        return {
            int(self.annotations.iloc[i, self.target_idx]): self.annotations.iloc[i, self.classname_idx]
            for i in range(len(self.annotations))
        }

    def get_category_to_target(self) -> Dict[str, int]:
        return {
            self.annotations.iloc[i, self.classname_idx]: int(self.annotations.iloc[i, self.target_idx])
            for i in range(len(self.annotations))
        }

    def load_folds(self) -> List[Example]:
        data = []
        for index in range(self.annotations.shape[0]):
            if self.annotations.iloc[index, self.fold_idx] not in self.folds:
                continue
            if self.dataset_name == "esc50":
                path = Path(self.directory) / "audio" / self.annotations.iloc[index, 0]
            else:
                path = (
                    Path(self.directory)
                    / "audio"
                    / f"fold{self.annotations.iloc[index, self.fold_idx]}"
                    / self.annotations.iloc[index, 0]
                )
            waveform, sample_rate = torchaudio.load(path)  # noqa
            waveform = remove_silence(waveform, self.max_length)
            # waveform = segment_waveform(waveform, self.max_length, self.overlap)
            label = torch.as_tensor(self.annotations.iloc[index, self.target_idx])
            label = torch.nn.functional.one_hot(label, num_classes=self.n_classes).float()
            data.append({"audio": waveform, "label": label})

        return data

    def as_data_loader(self) -> DataLoader:
        return DataLoader(self, batch_size=self.batch_size, shuffle=(not self.validation))
