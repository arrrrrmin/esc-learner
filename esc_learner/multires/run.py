import argparse
import logging
from pathlib import Path

from torch import nn, optim

from esc_learner import utils
from esc_learner.multires import configs
from esc_learner.envnet.dataset import Folds
from esc_learner.envnet.learner import Learner, Validator
from esc_learner.multires.model import MultiResCnn


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def multires_assets(config: argparse.Namespace):
    model = MultiResCnn(config.n_classes)
    utils.init_dropout(model, p=0.5)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.schedule, gamma=config.lr_gamma)

    return model, loss_fn, optimizer, scheduler


def main():
    config = configs.obtain_config()

    model, loss_fn, optimizer, scheduler = multires_assets(config)

    train_set = Folds(config.data, config.train_folds, validation=False, config=config)
    eval_set = Folds(config.data, config.eval_fold, validation=True, config=config)
    display_dataset_splits(train_set, eval_set)

    learner = Learner(model, loss_fn, optimizer, scheduler, train_set, eval_set, config)

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = learner.train(epoch)
        logger.info("Train Loss : %.6f - Train Acc : %.6f" % (train_loss, train_acc))

        # Validate every 10 epochs
        if epoch % config.eval_every == 0:
            eval_loss, eval_acc = learner.test(epoch)
            logger.info("Eval Loss : %.6f - Eval Acc : %.6f" % (eval_loss, eval_acc))

        if config.freeze_epoch == epoch:
            logger.info("Freezing params of feature extraction module, namely:")
            module = learner.model.freeze_feature_extraction()
            logger.info(module)

    best_model_checkpoint = learner.checkpoint_saver.checkpoints[0].name
    model_fp = Path(config.save) / best_model_checkpoint / f"{best_model_checkpoint}.model"
    model = MultiResCnn.load_state(config.n_classes, model_fp)

    validator = Validator(model, eval_set, config)
    validator.evaluate()


def display_dataset_splits(train_set: Folds, eval_set: Folds) -> None:
    logger.info("+++ Dataset split +++")
    logger.info("| Training length : {}".format(len(train_set)))
    logger.info("| Training folds : {}".format(train_set.folds))
    logger.info("| Validation length : {}".format(int(len(eval_set) / eval_set.num_val_crops)))
    logger.info("| Validation folds : {}".format(eval_set.folds))
    logger.info("++++++++++++++++++++++++++++++++")


# Example
#  poetry run python -m esc_learner.multires.run --dataset esc50 --data ./data/esc50-16000/ --save output/multires/
if __name__ == "__main__":
    main()
