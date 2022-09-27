import argparse
import shutil
import subprocess
from pathlib import Path
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def obtain_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", help="path to urban8k extracted directory")
    parser.add_argument("--save", default="data/")
    parser.add_argument("--sample_to", required=True, choices=[8000, 16000, 32000, 44100], type=int)

    config = parser.parse_args()

    save_dir = Path(config.save)
    save_dir.mkdir(exist_ok=True, parents=True)

    display_config(config)

    return config


def display_config(conf: argparse.Namespace) -> None:
    logger.info("++++++++++++++++++++++++++++++++")
    logger.info("| Urban8k config (prepare)")
    logger.info("++++++++++++++++++++++++++++++++")
    logger.info("| source : {}".format(conf.source))
    logger.info("| save : {}".format(conf.save))
    logger.info("| sample_to : {}".format(conf.sample_to))
    logger.info("++++++++++++++++++++++++++++++++")


def download_urban8k():
    conf = obtain_config()

    save_dir = Path(conf.save)
    save_dir.mkdir(exist_ok=True, parents=True)

    dst_dir = save_dir / f"urban8k-{conf.sample_to}/"
    dst_audio_dir = dst_dir / "audio/"
    dst_meta_dir = dst_dir / "meta/"
    dst_audio_dir.mkdir(exist_ok=True, parents=True)
    dst_meta_dir.mkdir(exist_ok=True, parents=True)

    audio_dir = Path(conf.source) / "audio/"
    meta_dir = Path(conf.source) / "metadata/"

    folds = sorted(audio_dir.glob("fold*/"), key=lambda d: int(d.stem.strip("fold")))
    for fold in folds:
        dst_fold = dst_audio_dir / fold.stem
        dst_fold.mkdir(parents=True, exist_ok=True)
        for fold_file in sorted(list(fold.glob("*.wav"))):
            dst_file = dst_fold / f"{fold_file.stem}.wav"
            subprocess.call(
                f"ffmpeg -i {fold_file} -ac 1 -ar {conf.sample_to} -loglevel error -y {dst_file}", shell=True
            )
    shutil.copy(meta_dir / "UrbanSound8K.csv", dst_meta_dir)
    print("Dataset preparation finished...")


if __name__ == "__main__":
    download_urban8k()
