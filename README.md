## How to use
For full documentation, visit [ginlov.github.io/cvrunner](https://ginlov.github.io/cvrunner).

### Logger
Set up these environment variables:
- WANDB_API_KEY
- CUBLAS_WORKSPACE_CONFIG
- PYTHONHASHSEED
- DOCKERHUB_USERNAME

You can take a look at the `.env.example` file.

### Runner
The `Runner` class maintains the status of a training job, including model state dict, optimizer, validation metrics, etc. It keeps the experiment as an attribute.

By default, all jobs use `DistributedTrainRunner` to support multi-GPU training.

**Customize runner:**  
You can create your own runner to fit your needs. Common functions to modify are:
- `run`
- `train_epoch_start`
- `train_epoch`
- `train_epoch_end`
- `val_epoch_start`
- `val_epoch`
- `val_epoch_end`
- `checkpoint`

### Experiment
The `Experiment` class defines the configuration for your experiment (e.g., number of epochs, batch size, model and dataset builders). This class is stateless. It also defines which runner to use via the `runner_cls()` function.

### Test

Run experiment:

**On local:**
```bash
cvrunner -e tests/test_runner.py -l
```
**In Docker container:**
```bash
cvrunner -e test/test_generator/mnist_components.py --target_image test_cvrunner --build
```
**On Kubernetes:**
```bash
cvrunner -e test/test_generator/mnist_components.py --target_image test_cvrunner --build --k8s
```

