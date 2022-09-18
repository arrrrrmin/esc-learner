import argparse
import json
import logging
import uuid
from pathlib import Path


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def obtain_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Abdoli learner script")

    parser.add_argument("--dataset", required=True, choices=["esc50"])
    parser.add_argument("--data", required=True, help="Path to dataset")
    parser.add_argument("--eval_fold", default=4, type=int, help="Fold for testing (excluded from training)")
    parser.add_argument("--save", default="None", help="Directory to save the results")

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01, help="Basic learning rate")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--sr", type=float, default=16000)
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--window_fn", type=str, choices=["rectangular", "hamming"], default="rectangular")
    parser.add_argument("--aggregator", type=str, choices=["majority_vote", "sum_rule"], default="majority_vote")

    config = parser.parse_args()

    # Setting defaults
    config.model = "abdolinet"
    if config.dataset == "esc50":
        config.n_classes = 50
        config.n_folds = 5
    config.segment_length = int(config.sr * config.length)
    config.train_folds = [i for i in range(1, config.n_folds + 1) if i != config.eval_fold]

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
    logger.info("| eval_fold : {}".format(conf.eval_fold))
    logger.info("| model : {}".format(conf.model))
    logger.info("| epochs : {}".format(conf.epochs))
    logger.info("| lr : {}".format(conf.lr))
    logger.info("| batch_size : {}".format(conf.batch_size))
    logger.info("| keep_n : {}".format(conf.keep_n))
    logger.info("| sample_rate : {}".format(conf.sr))
    logger.info("| length : {}".format(conf.lengt))
    logger.info("| overlap : {}".format(conf.overlap))
    logger.info("| aggregator : {}".format(conf.aggregator))
    logger.info("++++++++++++++++++++++++++++++++")
