import logging
from dataclasses import dataclass
import json
import random
import shutil
import string
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def msle(p: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return torch.square(torch.log(a + 1) - torch.log(p + 1)).sum()


def mean_squard_logarithmic_loss(p: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return (torch.log(torch.div(p + 1, a + 1)) ** 2).sum()


def count_correct_preds(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.eq(torch.argmax(p, dim=-1), torch.argmax(y, dim=-1)).float().sum()


def init_dropout(model: nn.Module, p: float):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = p


@dataclass
class Checkpoint:
    name: str
    epoch: int
    eval_acc: float


class CheckpointSaver:
    def __init__(self, keep_n: int, save_to: str):
        self.keep_n = keep_n
        self.save_to = save_to
        self.checkpoints: List[Checkpoint] = []

    def save_candidate(
        self, epoch: int, eval_acc: float, model: nn.Module, history: Dict[str, Dict[int, float]]
    ) -> None:
        # Save history regardless of model performance
        self.save_history_and_plots(history)

        if len(self.checkpoints) == self.keep_n and eval_acc <= self.checkpoints[-1].eval_acc:
            return

        if len(self.checkpoints) == self.keep_n:
            weakest = self.checkpoints.pop(-1)
            shutil.rmtree(str(Path(self.save_to) / weakest.name))

        checkpoint_name = self.save_checkpoint(model)
        self.checkpoints.append(Checkpoint(checkpoint_name, epoch, eval_acc))
        self.checkpoints = list(sorted(self.checkpoints, key=lambda m: m.eval_acc, reverse=True))
        self.save_overview()

        self.display_current_best()

    def save_history_and_plots(self, history: Dict[str, Dict[int, float]]) -> None:
        def save_plot(name: str):
            if not len(history[name].items()) >= 0:
                return
            d = np.array(list(history[name].items()), dtype=float)
            x, y = d[:, 0], d[:, 1]
            plt.plot(x, y, label=name)

        json.dump(history, (Path(self.save_to) / "history.json").open("w"), indent=4)
        output_dir = Path(self.save_to)
        for history_name in [k for k in history.keys() if "loss" in k]:
            save_plot(history_name)
        plt.legend()
        plt.savefig(output_dir / "losses.png")
        plt.close()
        for history_name in [k for k in history.keys() if "acc" in k]:
            save_plot(history_name)
        plt.legend()
        plt.savefig(output_dir / "accuracies.png")
        plt.close()

    def save_checkpoint(self, model: nn.Module, final: bool = False) -> str:
        checkpoint_name = "".join(random.sample(string.ascii_lowercase + string.digits, 10))
        if final:
            checkpoint_name = "final"
        output_dir = Path(self.save_to) / checkpoint_name
        output_dir.mkdir(parents=True)

        torch.save(model.state_dict(), output_dir / f"{checkpoint_name}.model")
        logger.info(f"Checkpoint written to '{output_dir}'")
        return checkpoint_name

    def save_overview(self) -> None:
        json.dump(self.as_dict(), (Path(self.save_to) / "checkpoints.json").open("w"), indent=4)

    def as_dict(self) -> Dict:
        return {cp.name: cp.__dict__ for cp in self.checkpoints}

    def display_current_best(self) -> None:
        logger.info("+++ Current best checkpoints  +++")
        overview = self.as_dict()
        for name in overview.keys():
            logger.info("+++ {}  +++".format(name))
            for k, v in overview[name].items():
                logger.info("| {} : {}".format(k, v))
        logger.info("++++++++++++++++++++++++++++++++")
