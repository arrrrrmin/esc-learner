import json
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.nn import functional as f
from tqdm import tqdm

from esc_learner import utils

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Learner:
    def __init__(self, model, loss_fn, optimizer, train_set, eval_set, config):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_set = train_set.as_data_loader()
        self.eval_set = eval_set.as_data_loader()
        self.config = config
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device=self.device)
        self.history = {"loss": {}, "acc": {}, "eval_loss": {}, "eval_acc": {}}
        self.checkpoint_saver = utils.CheckpointSaver(config.keep_n, save_to=config.save)

    def train(self, epoch: int) -> (torch.Tensor, torch.Tensor):
        self.model.train()
        logger.info(f"Train epoch: {epoch}")
        loss, acc = torch.tensor(0.0).to(self.device), torch.tensor(0.0).to(self.device)
        step = 0

        with tqdm(total=len(self.train_set), file=sys.stdout) as pbar:
            for batch in self.train_set:
                x_batch, y_batch = batch["audio"].to(self.device), batch["label"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model.forward(x_batch)
                train_loss = self.loss_fn(outputs, y_batch)
                train_loss.backward()
                self.optimizer.step()

                outputs = f.softmax(outputs, dim=-1)
                loss += train_loss
                acc += utils.count_correct_preds(outputs, y_batch) / outputs.size(0)
                step += 1
                pbar.update(1)

        loss = loss.item() / step
        acc = acc.item() / step
        self.history["loss"][epoch] = loss
        self.history["acc"][epoch] = acc

        return loss, acc

    @torch.no_grad()
    def test(self, epoch: int) -> (torch.Tensor, torch.Tensor):
        self.model.eval()
        loss, acc = torch.tensor(0.0).to(self.device), torch.tensor(0.0).to(self.device)
        step = 0

        with tqdm(total=len(self.eval_set), file=sys.stdout) as pbar:
            for batch in self.eval_set:
                x_batch, y_batch = batch["audio"].to(self.device), batch["label"].to(self.device)

                outputs = torch.mean(self.model.forward(x_batch), dim=0)
                loss += self.loss_fn(outputs, y_batch[0]).float()
                outputs = f.softmax(outputs, dim=-1)
                # No need to divide by batch_size, since we average the batch preds
                acc += utils.count_correct_preds(outputs, y_batch[0])
                step += 1
                pbar.update(1)

        loss = loss.item() / step
        acc = acc.item() / step
        self.history["eval_loss"][epoch] = loss
        self.history["eval_acc"][epoch] = acc

        self.checkpoint_saver.save_candidate(epoch, acc, self.model, self.history)
        return loss, acc

    def save_last_epoch(self) -> str:
        return self.checkpoint_saver.save_checkpoint(self.model, True)


class Validator:
    def __init__(self, model, eval_set, config):
        self.model = model
        self.eval_set = eval_set.as_data_loader()
        self.config = config
        self.target_to_cats = self.eval_set.dataset.get_target_to_category()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

    def evaluate(self) -> None:
        self.model.eval()
        acc = torch.tensor(0.0).to(self.device)
        step = 0
        preds = []
        gts = []
        for batch in self.eval_set:
            x_batch, y_batch = batch["audio"].to(self.device), batch["label"].to(self.device)

            output = torch.mean(self.model.predict(x_batch), dim=0)
            preds.append(torch.argmax(output, dim=-1).cpu())
            gts.append(torch.argmax(y_batch[0], dim=-1).cpu())
            acc += utils.count_correct_preds(output, y_batch[0])
            step += 1

        acc = acc.item() / step
        conf_mat = self.build_confusion_matrix(preds, gts)

        (Path(self.config.save) / "eval").mkdir(parents=True, exist_ok=True)
        conf_mat.to_csv(Path(self.config.save) / "eval" / "confusion_matrix.csv")
        json.dump({"acc": acc}, (Path(self.config.save) / "eval" / "result.json").open("w"))

    def build_confusion_matrix(self, p: List[torch.Tensor], a: List[torch.Tensor]) -> pd.DataFrame:
        return pd.DataFrame(
            confusion_matrix(p, a),
            columns=[self.target_to_cats[i] for i in range(self.config.n_classes)],
            index=[self.target_to_cats[i] for i in range(self.config.n_classes)],
        )
