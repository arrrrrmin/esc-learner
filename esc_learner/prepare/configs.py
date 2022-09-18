import argparse
import logging
from pathlib import Path

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def obtain_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--save", default="data/")
    parser.add_argument("--sample_to", required=True, choices=[16000, 44100], type=int)

    config = parser.parse_args()

    save_dir = Path(config.save)
    save_dir.mkdir(exist_ok=True, parents=True)

    display_config(config)

    return config


def display_config(conf: argparse.Namespace) -> None:
    logger.info("++++++++++++++++++++++++++++++++")
    logger.info("| ESC config (download & resample)")
    logger.info("++++++++++++++++++++++++++++++++")
    logger.info("| save : {}".format(conf.save))
    logger.info("| sample_to : {}".format(conf.sample_to))
    logger.info("++++++++++++++++++++++++++++++++")
