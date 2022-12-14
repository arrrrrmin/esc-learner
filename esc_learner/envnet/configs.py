import argparse
import json
import logging
import uuid
from pathlib import Path


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def obtain_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Envnet learner script")

    parser.add_argument("--dataset", required=True, choices=["esc50"])
    parser.add_argument("--model", required=True, choices=["envnet", "envnetv2"])
    parser.add_argument("--data", required=True, help="Path to dataset")
    parser.add_argument("--eval_folds", type=int, nargs="*", default=[4], help="Fold for testing")
    parser.add_argument("--save", default="None", help="Directory to save results")
    parser.add_argument("--keep_n", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01, help="Basic learning rate")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="Learning rate gamma adjustment")
    parser.add_argument("--schedule", type=int, nargs="*", default=[80, 100, 120], help="Steps for lr")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--freeze_epoch", type=int, default=-1, help="At this epoch only train classification head")

    parser.add_argument("--crops", type=int, default=10)
    parser.add_argument("--bc", action="store_true")

    config = parser.parse_args()

    if config.dataset == "esc50":
        config.n_classes = 50
        config.n_folds = 5

    config.fs = 16000
    config.max_length = 24014
    if config.model == "envnetv2":
        config.fs = 44100
        config.max_length = 66650
    config.train_folds = [i for i in range(1, config.n_folds + 1) if i not in config.eval_folds]

    if config.save == "None":
        config.save = "output/"
    config.save = str(Path(config.save) / uuid.uuid4().hex)
    Path(config.save).mkdir(parents=True)
    json.dump(config.__dict__, (Path(config.save) / "config.json").open("w"), indent=4)

    display_config(config)

    return config


def display_config(conf: argparse.Namespace) -> None:
    logger.info("++++++++++++++++++++++++++++++++")
    logger.info("| ESC Params")
    logger.info("++++++++++++++++++++++++++++++++")
    logger.info("| dataset : {}".format(conf.dataset))
    logger.info("| save : {}".format(conf.save))
    logger.info("| eval_folds : {}".format(conf.eval_folds))
    logger.info("| model : {}".format(conf.model))
    logger.info("| bc : {}".format(conf.bc))
    logger.info("| epochs : {}".format(conf.epochs))
    logger.info("| lr : {}".format(conf.lr))
    logger.info("| lr_gamma : {}".format(conf.lr_gamma))
    logger.info("| schedule : {}".format(conf.schedule))
    logger.info("| batch_size : {}".format(conf.batch_size))
    logger.info("| keep_n : {}".format(conf.keep_n))
    logger.info("++++++++++++++++++++++++++++++++")
