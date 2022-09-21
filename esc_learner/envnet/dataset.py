import argparse
import random
from pathlib import Path
from typing import List, TypedDict, Union, Dict

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from esc_learner.envnet import bc


def remove_silence(waveform: torch.Tensor, max_length: int) -> torch.Tensor:
    waveform = waveform[:, waveform.nonzero().min() : waveform.nonzero().max()]
    pad_width = max_length // 2
    return torch.nn.functional.pad(waveform, (pad_width, pad_width), mode="constant", value=0)


def random_selection(waveform: torch.Tensor, max_length: int) -> torch.Tensor:
    start_selected = random.randint(0, waveform.shape[-1] - max_length)
    return waveform[:, start_selected : (start_selected + max_length)]


def crop_validation(waveform: torch.Tensor, max_length: int, num_crops: int) -> List[torch.Tensor]:
    stride = (waveform.size(1) - max_length) // (num_crops - 1)
    return [waveform[:, stride * n : stride * n + max_length] for n in range(num_crops)]


class Example(TypedDict):

    audio: torch.Tensor
    label: torch.Tensor


class Folds(Dataset):
    def __init__(self, directory: str, folds: Union[List[int], int], validation: bool, config: argparse.Namespace):
        self.directory = directory
        self.folds = folds if isinstance(folds, list) else [folds]
        self.validation = validation

        self.bc_training = config.bc
        self.fs = config.fs

        self.max_length = config.max_length
        self.num_val_crops = config.crops
        self.batch_size = config.batch_size if not self.validation else self.num_val_crops
        self.n_classes = config.n_classes

        self.annotations = pd.read_csv(Path(self.directory) / "meta" / "esc50.csv", delimiter=",")
        self.annotations = self.annotations[self.annotations.fold.map(lambda fold: fold in self.folds)]
        self.data = self.load_folds()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Example:
        if not self.validation:
            if self.bc_training:
                return self.mix_samples(index)

            waveform = remove_silence(self.data[index]["audio"], self.max_length)
            return Example(
                audio=random_selection(waveform, self.max_length),
                label=self.data[index]["label"],
            )

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
            label = torch.as_tensor(self.annotations.iloc[index, 2])
            label = torch.nn.functional.one_hot(label, num_classes=self.n_classes).float()
            if self.validation:
                # When we validate we need to set batch_size = num_val_crops
                waveform = crop_validation(waveform, self.max_length, self.num_val_crops)
                data.extend([{"audio": w, "label": label} for w in waveform])
            else:
                data.append({"audio": waveform, "label": label})

        return data

    def get_conter_sample(self, example: Example) -> Example:
        while True:
            mixable_example = self.data[random.randint(0, len(self.data) - 1)]
            if not torch.equal(example["label"], mixable_example["label"]):
                break
        return mixable_example

    def mix_samples(self, index: int):
        r = np.array(random.random())
        example_a = self.data[index]
        example_b = self.get_conter_sample(example_a)
        sound_a = example_a["audio"]
        sound_b = example_b["audio"]
        sound = random_selection(bc.mix_samples(sound_a, sound_b, r, self.fs).astype(np.float32), self.max_length)
        eye = np.eye(self.n_classes)
        l_a = example_a["label"].argmax().item()
        l_b = example_b["label"].argmax().item()
        label = (eye[l_a] * r + eye[l_b] * (1 - r)).astype(np.float32)
        return {"audio": sound, "label": label}

    def as_data_loader(self) -> DataLoader:
        return DataLoader(self, batch_size=self.batch_size, shuffle=(not self.validation))
