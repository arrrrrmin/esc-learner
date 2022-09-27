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

    parser.add_argument("--dataset", required=True, choices=["urbansound8k"])
    parser.add_argument("--data", required=True, help="Path to dataset")
    parser.add_argument("--eval_folds", type=int, nargs="*", default=[10], help="Fold for testing")
    parser.add_argument("--save", default="None", help="Directory to save results")
    parser.add_argument("--keep_n", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001, help="Basic learning rate")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--m", type=int, choices=[3, 5, 11, 18], help="Model architecture config")

    config = parser.parse_args()

    if config.dataset == "urbansound8k":
        config.n_classes = 10
        config.n_folds = 10

    config.fs = 8000
    config.max_length = 32079
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
    logger.info("| epochs : {}".format(conf.epochs))
    logger.info("| lr : {}".format(conf.lr))
    logger.info("| batch_size : {}".format(conf.batch_size))
    logger.info("| keep_n : {}".format(conf.keep_n))
    logger.info("++++++++++++++++++++++++++++++++")
