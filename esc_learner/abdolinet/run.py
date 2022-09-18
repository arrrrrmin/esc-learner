import argparse
import logging
from pathlib import Path

from torch import optim

from esc_learner import utils
from esc_learner.abdolinet import configs
from esc_learner.abdolinet.dataset import Folds
from esc_learner.abdolinet.learner import Learner, Validator
from esc_learner.abdolinet.model import AbdoliNet
from esc_learner.envnet.model import EnvNet


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def envnet_assets(config: argparse.Namespace):
    model = AbdoliNet(config.n_classes, config.aggregator)
    loss_fn = utils.mean_squard_logarithmic_loss
    # rho=0.9, eps=1e-06
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return model, loss_fn, optimizer


def main():
    config = configs.obtain_config()

    model, loss_fn, optimizer = envnet_assets(config)

    train_set = Folds(config.data, config.train_folds, validation=False, config=config)
    eval_set = Folds(config.data, config.eval_fold, validation=True, config=config)
    display_dataset_splits(train_set, eval_set)

    learner = Learner(model, loss_fn, optimizer, train_set, eval_set, config)

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = learner.train(epoch)
        logger.info("Train Loss : %.6f - Train Acc : %.6f" % (train_loss, train_acc))

        # Validate every 10 epochs
        if epoch % config.eval_every == 0:
            eval_loss, eval_acc = learner.test(epoch)
            logger.info("Eval Loss : %.6f - Eval Acc : %.6f" % (eval_loss, eval_acc))

    best_model_checkpoint = learner.checkpoint_saver.checkpoints[0].name
    model_fp = Path(config.save) / best_model_checkpoint / f"{best_model_checkpoint}.model"
    model = EnvNet.load_state(config.n_classes, model_fp)

    validator = Validator(model, eval_set, config)
    validator.evaluate()


def display_dataset_splits(train_set: Folds, eval_set: Folds) -> None:
    logger.info("+++ Dataset split +++")
    logger.info("| Training length : {}".format(len(train_set)))
    logger.info("| Training folds : {}".format(train_set.folds))
    logger.info("| Validation length : {}".format(len(eval_set)))
    logger.info("| Validation folds : {}".format(eval_set.folds))
    logger.info("++++++++++++++++++++++++++++++++")


if __name__ == "__main__":
    main()
