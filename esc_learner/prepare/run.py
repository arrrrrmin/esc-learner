import argparse
import os
import shutil
import subprocess
from pathlib import Path

import requests


def download_esc_50():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save", default="data/")
    parser.add_argument("--sample_to", required=True, choices=[16000, 44100], type=int)

    conf = parser.parse_args()

    save_dir = Path(conf.save)
    save_dir.mkdir(exist_ok=True, parents=True)

    response = requests.get("https://github.com/karoldvl/ESC-50/archive/master.zip")
    (save_dir / "master.zip").write_bytes(response.content)
    subprocess.call(f"unzip -q -d {save_dir} {save_dir / 'master.zip'}", shell=True)
    os.remove(save_dir / "master.zip")

    dst_dir = save_dir / f"esc50-{conf.sample_to}/"
    dst_audio_dir = dst_dir / "audio/"
    dst_meta_dir = dst_dir / "meta/"
    dst_audio_dir.mkdir(exist_ok=True, parents=True)
    dst_meta_dir.mkdir(exist_ok=True, parents=True)

    audio_dir = save_dir / "ESC-50-master" / "audio/"
    meta_dir = save_dir / "ESC-50-master" / "meta/"

    for src_file in sorted(audio_dir.glob("*.wav")):
        dst_file = dst_audio_dir / f"{src_file.stem}.wav"
        if not dst_file.exists() or conf.sample_to != 44100:
            subprocess.call(
                f"ffmpeg -i {src_file} -ac 1 -ar {conf.sample_to} -loglevel error -y {dst_file}", shell=True
            )

    shutil.copy(meta_dir / "esc50.csv", dst_meta_dir)
    shutil.rmtree(save_dir / "ESC-50-master")


if __name__ == "__main__":
    download_esc_50()
