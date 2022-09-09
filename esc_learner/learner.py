import logging
import sys

import torch
from tqdm import tqdm

from esc_learner.utils import CheckpointSaver

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Learner:
    def __init__(self, model, loss_fn, optimizer, scheduler, train_set, eval_set, config):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_set = train_set
        self.eval_set = eval_set
        self.config = config
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device=self.device)
        self.history = {"loss": {}, "acc": {}, "eval_loss": {}, "eval_acc": {}}
        self.best_models = []
        self.checkpoint_saver = CheckpointSaver(config.keep_n, save_to=config.save)

    def train(self, epoch: int) -> None:
        self.model.train()
        logger.info(f"Train epoch: {epoch}")
        with tqdm(total=len(self.train_set), file=sys.stdout) as pbar:
            for step, batch in enumerate(self.train_set):
                x_batch, y_batch = batch["audio"].to(self.device), batch["label"].to(self.device)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.model.forward(x_batch)
                    loss = self.loss_fn(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                pbar.update(1)

    @torch.no_grad()
    def val_training(self, epoch: int) -> (torch.Tensor, torch.Tensor):
        self.model.eval()
        loss = torch.tensor(0.0).to(self.device)
        acc = torch.tensor(0.0).to(self.device)
        step = 0

        for step, batch in enumerate(self.train_set):
            x_batch, y_batch = batch["audio"].to(self.device), batch["label"].to(self.device)
            outputs = self.model.predict(x_batch)
            loss += self.loss_fn(outputs, y_batch)
            acc += torch.eq(torch.argmax(outputs, dim=-1), torch.argmax(y_batch, dim=-1)).float().sum() / x_batch.size(
                0
            )
            step += 1

        loss = loss.item() / step
        acc = acc.item() / step
        self.history["loss"][epoch] = loss
        self.history["acc"][epoch] = acc
        return loss, acc

    @torch.no_grad()
    def val_testing(self, epoch: int) -> (torch.Tensor, torch.Tensor):
        self.model.eval()
        loss = torch.tensor(0.0).to(self.device)
        acc = torch.tensor(0.0).to(self.device)
        step = 0

        for step, batch in enumerate(self.eval_set):
            x_batch, y_batch = batch["audio"].to(self.device), batch["label"].to(self.device)
            outputs = torch.mean(self.model.predict(x_batch), dim=0)
            loss += self.loss_fn(outputs, y_batch[0]).float()
            # No need to divide by batch_size, since we mean the batch anyway
            acc += torch.eq(torch.argmax(outputs, dim=-1), torch.argmax(y_batch[0], dim=-1)).float().sum()
            step += 1
        loss = loss.item() / step
        acc = acc.item() / step
        self.history["eval_loss"][epoch] = loss
        self.history["eval_acc"][epoch] = acc
        self.checkpoint_saver.save_candidate(epoch, acc, self.model, self.history)
        return loss, acc
