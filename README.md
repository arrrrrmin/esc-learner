# esc-learner

A project for getting started with end-to-end convolutional neural networks in
environmental sound classification (ESC) domain.

ESC is a fundamental task in the audio domain, working mainly with three datasets.
ESC-10, ESC-50 and Urbansound8k. ESC is the basis for other tasks for example
Sound Event Detection (SED).

End-to-end network take raw waveforms as input tensors and stack convolutional layers
in different ways to learn not only the classification task but also how to extrac
accoustic features while training.

> This project is intended for self-educational purposes, but all assets can be accessed/replicated.
> Models are implemented with pytorch trained and tested on a mid-tier GPU.

## Dependencies

| Dependency   | Version |
|--------------|---------|
| python       | ^3.8    |
| torch        | 1.12.0  |
| torchaudio   | 0.12.0  |
| numpy        | 1.23.2  |
| requests     | 2.28.1  |
| pandas       | 1.4.4   |
| tqdm         | 4.64.1  |
| scikit-learn | 1.1.2   |

## Available Models

| Model                                            | Authors         | Publication                                                                 |
|--------------------------------------------------|-----------------|-----------------------------------------------------------------------------|
| [`wavemsnet`](esc_learner/wavemsnet/__init__.py) | Boqing et.al.   | Learning Environmental Sounds with Multi-scale Convolutional Neural Network |
| [`envnet`](esc_learner/envnet/__init__.py)       | Tokozume et.al. | Learning Environmental Sounds with End-to-End Convolutional Neural Networks |
| [`envnetv2`](esc_learner/envnet/__init__.py)     | Tokozume et.al. | Learning from Between-class Examples for Deep Sound Recognition             |
| [`wavemxnet`](esc_learner/wavemxnet/__init__.py) | Dai et.al.      | Very Deep Convolutional Neural Networks for Raw Waveforms                   |

## Preparing Data

Every used dataset ([`ESC-50`](https://github.com/karolpiczak/ESC-50) and
[`UrbanSound8k`](https://urbansounddataset.weebly.com/urbansound8k.html)) is prepared using the [`prepare`](esc_learner/prepare/) package.
The preparation script for `ESC-50` downloads the dataset whereas `UrbanSound8k` has to be downloaded manually.
For `ESC-50`:
```
$ poetry run python -m esc_learner.prepare.esc --help
usage: esc.py [-h] [--save SAVE] --sample_to {16000,44100}

optional arguments:
  -h, --help            show this help message and exit
  --save SAVE
  --sample_to {16000,44100}
```
For `UrbanSound8k`:
```
poetry run python -m esc_learner.prepare.urban8k --help
usage: urban8k.py [-h] [--source SOURCE] [--save SAVE] --sample_to {8000,16000,32000,44100}

optional arguments:
  -h, --help            show this help message and exit
  --source SOURCE       path to urban8k extracted directory
  --save SAVE
  --sample_to {8000,16000,32000,44100}
```

## Dataset loaders

Some models use their own `dataset.py` located in the corresponding package `esc_learner/<model_name>/`.
In case of `wavemsnet`, this model uses the data loader of `envnet`.

## Train models

Each model has an own `run.py` to execute training with a certain configuration, which is available in a seperate
`config.py`. The output assets such as the config, model file, checkpoints, plots and meta-data json files are
written to `--save`. Model architectures are available at `esc_learner/<model_name>/model.py`.

## Access trained models

| Model           | Url                                                                                                     |
|-----------------|---------------------------------------------------------------------------------------------------------|
| envnet          | https://esc-learner-assets.s3.eu-central-1.amazonaws.com/envnet/3ea7995342f14062ab9816014010b287.zip    |
| wavemsnet       | https://esc-learner-assets.eu-central-1.amazonaws.com/wavemsnet/9a3f9b0ecba94a61b67572dec1a1aafd.zip    |
| wavemxnet (m3)  | https://esc-learner-assets.s3.eu-central-1.amazonaws.com/wavemxnet/54d6681d66b5417ea1dcd5a7332d75d6.zip |
| wavemxnet (m5)  | https://esc-learner-assets.s3.eu-central-1.amazonaws.com/wavemxnet/dd6569688fb847d6a5edb4548fcb4930.zip |
| wavemxnet (m11) | https://esc-learner-assets.s3.eu-central-1.amazonaws.com/wavemxnet/e59195cf3d3a41c896ccd8f990cf1afe.zip |
| wavemxnet (m18) | https://esc-learner-assets.s3.eu-central-1.amazonaws.com/wavemxnet/be0a6daefcc04787a959c3c1b7c1b0cf.zip |

Todo: `envnetv2`

Each directory has the same structure:

```
$ tree output/wavemxnet
output/wavemxnet
├── [4.0K]  e59195cf3d3a41c896ccd8f990cf1afe
│   ├── [4.0K]  8k2px1vh6f
│   │   └── [6.9M]  8k2px1vh6f.model
│   ├── [ 26K]  accuracies.png
│   ├── [ 236]  checkpoints.json
│   ├── [ 517]  config.json
│   ├── [4.0K]  en8952kyuj
│   │   └── [6.9M]  en8952kyuj.model
│   ├── [4.0K]  eval
│   │   ├── [ 447]  confusion_matrix.csv
│   │   └── [  27]  result.json
│   ├── [8.1K]  history.json
│   └── [ 24K]  losses.png
```

If you open `config.json` of the above chosen `wavemxnet/e59195cf3d3a41c896ccd8f990cf1afe/` you'll see what
arguments been used in training. In this case:

```json
{
    "dataset": "urbansound8k",
    "data": "./data/urban8k-8000/",
    "eval_folds": [10],
    "save": "output/wavemxnet/e59195cf3d3a41c896ccd8f990cf1afe",
    "keep_n": 2,
    "epochs": 100,
    "eval_every": 5,
    "lr": 0.0001,
    "batch_size": 64,
    "weight_decay": 0.0001,
    "m": 11,
    "n_classes": 10,
    "n_folds": 10,
    "fs": 8000,
    "max_length": 32079,
    "train_folds": [1, 2, 3, 4, 5, 6, 7, 8, 9]
}
```

Depending on the model you want to inspect this config can vary. For example some use scheduling, so these
provide a schedule config with steps and learning gamma.

## Evaluate models

For evaluation run `esc_learner/<model_name>/eval.py`. This takes only one argument `--run_dir` which is
the path to the run directory. For the above example use `--run_dir wavemxnet/e59195cf3d3a41c896ccd8f990cf1afe/`.

Running `eval.py` will create a folder in the training output dir (`--run_dir`) and create two files
* `results.json`
* `confusion_matrix.csv`

This will look like this for models trained on UrbanSound8k:

|                  | air_conditioner | car_horn | children_playing | dog_bark | drilling | engine_idling | gun_shot | jackhammer | siren | street_music |
|------------------|-----------------|----------|------------------|----------|----------|---------------|----------|------------|-------|--------------|
| air_conditioner  | 40              | 0        | 4                | 2        | 0        | 5             | 0        | 0          | 0     | 4            |
| car_horn         | 0               | 26       | 0                | 2        | 1        | 0             | 0        | 0          | 0     | 1            |
| children_playing | 15              | 1        | 61               | 12       | 1        | 6             | 1        | 0          | 19    | 10           |
| dog_bark         | 4               | 2        | 2                | 73       | 0        | 1             | 0        | 0          | 10    | 0            |
| drilling         | 9               | 2        | 7                | 3        | 79       | 12            | 0        | 21         | 1     | 6            |
| engine_idling    | 5               | 0        | 0                | 0        | 0        | 58            | 0        | 2          | 0     | 1            |
| gun_shot         | 0               | 0        | 0                | 1        | 0        | 0             | 30       | 0          | 0     | 0            |
| jackhammer       | 1               | 0        | 0                | 1        | 1        | 5             | 0        | 72         | 0     | 0            |
| siren            | 1               | 0        | 8                | 0        | 12       | 6             | 1        | 0          | 49    | 0            |
| street_music     | 25              | 2        | 18               | 6        | 6        | 0             | 0        | 1          | 4     | 78           |
