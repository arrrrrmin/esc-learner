import argparse
import logging

from torch import optim, nn

from esc_learner import configs, models
from esc_learner.dataset import Folds
from esc_learner.learner import Learner


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def envnet_assets(config: argparse.Namespace):
    if config.model != "envnet":
        raise ValueError("A --model with 'envnet' is required to learn EnvNetV1")

    model = models.model_archive[config.model](config.n_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=config.lr, nesterov=True, momentum=config.momentum, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.schedule, gamma=config.lr_gamma)

    return model, loss_fn, optimizer, scheduler


def main():
    config = configs.obtain_config()

    model, loss_fn, optimizer, scheduler = envnet_assets(config)

    train_set = Folds(config.data, config.train_folds, validation=False, config=config)
    eval_set = Folds(config.data, config.eval_fold, validation=True, config=config)
    display_dataset_splits(train_set, eval_set)

    learner = Learner(
        model, loss_fn, optimizer, scheduler, train_set.as_data_loader(), eval_set.as_data_loader(), config
    )

    for epoch in range(1, config.epochs + 1):
        learner.train(epoch)

        # Validate every 10 epochs
        if epoch % 10 == 0:
            train_loss, train_acc = learner.val_training(epoch)
            logger.info("TrainLoss : %.5f - TrainAcc : %.5f" % (train_loss, train_acc))
            eval_loss, eval_acc = learner.val_testing(epoch)
            logger.info("EvalLoss : %.5f - EvalAcc : %.5f" % (eval_loss, eval_acc))


def display_dataset_splits(train_set: Folds, eval_set: Folds) -> None:
    logger.info("+++ Dataset split +++")
    logger.info("| train_length : {}".format(len(train_set)))
    logger.info("| folds : {}".format(train_set.folds))
    logger.info("| eval_length : {}".format(len(eval_set)))
    logger.info("| folds : {}".format(eval_set.folds))
    logger.info("++++++++++++++++++++++++++++++++")


if __name__ == "__main__":
    main()
