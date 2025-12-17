# CVRunner

CVRunner simplifies and automates the setup for deep learning model training, making it easier to manage experiments, logging, and distributed training.

## Introduction

CVRunner is designed to streamline the process of training deep learning models by providing:

- Automated logging with Weights & Biases (wandb)
- Multi-GPU and distributed training support via `DistributedTrainRunner`
- Flexible configuration using a Python-based `Experiment` class
- Seamless operation on local machines, Docker containers, and Kubernetes clusters

## Features

- **Customizable Training Logic:** Easily extend or modify the training workflow by subclassing `DistributedTrainRunner`.
- **Experiment Configuration:** Define all experiment parameters (epochs, batch size, model, dataset, etc.) in a stateless `Experiment` class.
- **Integrated Logging:** Automatic experiment tracking with wandb.
- **Multi-Environment Support:** Run experiments locally, in Docker, or on Kubernetes with minimal changes.

## Getting Started

For full documentation, visit [ginlov.github.io/cvrunner](https://ginlov.github.io/cvrunner).

### Prerequisites

Set the following environment variables before running experiments:

- `WANDB_API_KEY`
- `CUBLAS_WORKSPACE_CONFIG`
- `PYTHONHASHSEED`
- `DOCKERHUB_USERNAME`

You can use the `.env.example` file as a template.

### Usage

#### Running an Experiment

**Locally:**
```bash
cvrunner -e tests/test_runner.py -l
```

**In a Docker Container:**
```bash
cvrunner -e test/test_generator/mnist_components.py --target_image test_cvrunner --build
```

**On Kubernetes:**
```bash
cvrunner -e test/test_generator/mnist_components.py --target_image test_cvrunner --build --k8s
```

### Customizing the Runner

The `Runner` class manages the state of a training job (model, optimizer, metrics, etc.) and holds a reference to the experiment configuration. By default, `DistributedTrainRunner` is used for multi-GPU support.

To customize training, subclass the runner and override methods such as:

- `run`
- `train_epoch_start`
- `train_epoch`
- `train_epoch_end`
- `val_epoch_start`
- `val_epoch`
- `val_epoch_end`
- `checkpoint`

### Defining an Experiment

The `Experiment` class specifies all configuration for your experiment and selects the runner via the `runner_cls()` method.

---

For more details and advanced usage, see the [documentation](https://ginlov.github.io/cvrunner).

