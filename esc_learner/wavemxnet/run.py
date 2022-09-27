import argparse
import logging
from pathlib import Path

from torch import nn, optim

from esc_learner import utils
from esc_learner.wavemxnet import configs
from esc_learner.wavemxnet.dataset import Folds
from esc_learner.wavemxnet.model import WaveMxNet
from esc_learner.wavemxnet.learner import Learner, Validator


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def mxnet_assets(config: argparse.Namespace):
    model = WaveMxNet(config.n_classes, config.m)
    utils.init_weights_xavier(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    return model, loss_fn, optimizer


def main():
    config = configs.obtain_config()

    model, loss_fn, optimizer = mxnet_assets(config)

    train_set = Folds(config.data, config.train_folds, validation=False, config=config)
    eval_set = Folds(config.data, config.eval_folds, validation=True, config=config)
    display_dataset_splits(train_set, eval_set)

    learner = Learner(model, loss_fn, optimizer, None, train_set, eval_set, config)

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = learner.train(epoch)
        logger.info("Train Loss : %.6f - Train Acc : %.6f" % (train_loss, train_acc))

        # Validate every 10 epochs
        if epoch % config.eval_every == 0:
            eval_loss, eval_acc = learner.test(epoch)
            logger.info("Eval Loss : %.6f - Eval Acc : %.6f" % (eval_loss, eval_acc))

    best_model_checkpoint = learner.checkpoint_saver.checkpoints[0].name
    model_fp = Path(config.save) / best_model_checkpoint / f"{best_model_checkpoint}.model"
    model = WaveMxNet.load_state(config.n_classes, config.m, model_fp)

    validator = Validator(model, eval_set, config)
    validator.evaluate()


def display_dataset_splits(train_set: Folds, eval_set: Folds) -> None:
    logger.info("+++ Dataset split +++")
    logger.info("| Training length : {}".format(len(train_set)))
    logger.info("| Training folds : {}".format(train_set.folds))
    logger.info("| Validation length : {}".format(len(eval_set)))
    logger.info("| Validation folds : {}".format(eval_set.folds))
    logger.info("++++++++++++++++++++++++++++++++")


# Example
#  poetry run python -m esc_learner.wavemxnet.run --dataset urbansound8k --data ./data/urban8k-8000/
#   --save output/wavemxnet/
if __name__ == "__main__":
    main()
