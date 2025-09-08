import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from cvrunner.experiment.experiment import BaseExperiment
from cvrunner.runner.runner import TrainRunner
from cvrunner.utils.logger import get_cv_logger

from tests.dummy_dataset import DummyDataset

logger = get_cv_logger()

# -----------------------------
# Dummy Experiment
# -----------------------------
class DummyExperiment(BaseExperiment):
    def __init__(self):
        # Nothing to keep stateful (your design goal)
        pass

    @property
    def runner_cls(self) -> object:
        return TrainRunner
    
    @property
    def batch_size(self) -> int:
        return 8

    @property
    def num_epochs(self) -> int:
        return 2

    @property
    def val_freq(self) -> int:
        return 1

    def build_dataset(self, partition: str) -> Dataset:
        return DummyDataset(size=50 if partition == "train" else 20)

    def build_model(self) -> torch.nn.Module:
        return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def build_dataloader(self, partition: str) -> DataLoader:
        return DataLoader(
            self.build_dataset(partition),
            batch_size=self.batch_size,
            shuffle=True
        )

    def build_optimizer_scheduler(self, model: torch.nn.Module):
        optimizer = Adam(model.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
        return optimizer, scheduler

    def build_loss_function(self):
        return nn.CrossEntropyLoss()

    def save_checkpoint(self):
        print("Checkpoint saved (dummy).")

    def load_checkpoint(self):
        print("Checkpoint loaded (dummy).")

    def train_epoch_start(self):
        print("Training epoch started.")

    def train_step(self, model, data_batch, loss_function, optimizer, lr_scheduler):
        inputs, labels = data_batch["inputs"], data_batch["labels"]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        logger.info(f"Loss value {loss.item()}")
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    def train_epoch_end(self):
        print("Training epoch ended.")

    def val_epoch_start(self):
        print("Validation epoch started.")

    def val_step(self, model, data_batch, loss_function, criterion):
        inputs, labels = data_batch["inputs"], data_batch["labels"]
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        return {"val_loss": loss.item()}

    def val_epoch_end(self):
        print("Validation epoch ended.")

def test_runner_trains_one_epoch():
    exp = DummyExperiment()
    runner = exp.runner_cls(exp)
    runner.run()
