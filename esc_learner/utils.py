import logging
from dataclasses import dataclass
import json
import random
import shutil
import string
from pathlib import Path
from typing import List, Dict

import torch
from torch import nn


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
        if len(self.checkpoints) == self.keep_n and eval_acc <= self.checkpoints[-1].eval_acc:
            return

        if len(self.checkpoints) == self.keep_n:
            weakest = self.checkpoints.pop(-1)
            shutil.rmtree(str(Path(self.save_to) / weakest.name))

        checkpoint_name = self.save_checkpoint(model, history)
        self.checkpoints.append(Checkpoint(checkpoint_name, epoch, eval_acc))
        self.checkpoints = list(sorted(self.checkpoints, key=lambda m: m.eval_acc, reverse=True))

        self.display_current_best()

    def save_checkpoint(self, model: nn.Module, history: Dict[str, Dict[int, float]]) -> str:
        checkpoint_name = "".join(random.sample(string.ascii_lowercase + string.digits, 10))
        output_dir = Path(self.save_to) / checkpoint_name
        output_dir.mkdir(parents=True)

        torch.save(model.state_dict(), output_dir / f"{checkpoint_name}.model")
        with (output_dir / f"{checkpoint_name}-history.json").open("w") as f:
            f.write(json.dumps(history, indent=4))

        logger.info(f"Checkpoint written to '{output_dir}'")
        return checkpoint_name

    def display_current_best(self) -> None:
        logger.info("+++ Current best checkpoint  +++")
        logger.info("| epoch : {}".format(self.checkpoints[0].epoch))
        logger.info("| name : {}".format(self.checkpoints[0].name))
        logger.info("| eval_acc : {}".format(self.checkpoints[0].eval_acc))
        logger.info("++++++++++++++++++++++++++++++++")
