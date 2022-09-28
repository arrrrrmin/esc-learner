import argparse
import json
import logging
from pathlib import Path

from esc_learner.wavemxnet.dataset import Folds
from esc_learner.wavemxnet.learner import Validator
from esc_learner.wavemxnet.model import WaveMxNet


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def eval_run(run_dir: str):
    run_dir = Path(run_dir)
    config = argparse.Namespace(**json.load((run_dir / "config.json").open("r")))
    checkpoints = json.load((run_dir / "checkpoints.json").open("r"))
    best_checkpoint = sorted(
        [checkpoints[cp_name] for cp_name in checkpoints.keys()], key=lambda x: x["eval_acc"], reverse=True
    )[0]

    logger.info(
        f"Best checkpoint for {run_dir} found at '{best_checkpoint['name']}' (epoch {best_checkpoint['epoch']})"
    )

    model_fp = run_dir / best_checkpoint["name"] / f"{best_checkpoint['name']}.model"
    model = WaveMxNet.load_state(config.n_classes, config.m, model_fp)
    eval_set = Folds(config.data, config.eval_folds, validation=True, config=config)

    validator = Validator(model, eval_set, config)
    validator.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WaveMxNet evaluator script")
    parser.add_argument("--run_dir", required=True)
    arguments = parser.parse_args()
    eval_run(arguments.run_dir)
