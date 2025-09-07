import torch

from abc import ABC, abstractmethod
from typing import Any, Tuple, Union, Dict

from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

DataBatch = Dict[str, Union[Any, torch.tensor]]

class BaseExperiment(ABC):
    """
    Template for the configuration.
    Every experiments should be stateless.
    """
    # TODO: find best practice to define runner class, prevent rounded import
    @property
    def runner_cls(self) -> object:
        return None
    
    @property
    def batch_size(self) -> int:
        return 0
    
    @property
    def num_epochs(self) -> int:
        return 0
    
    @property
    def val_freq(self) -> int:
        return 10
    
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def build_dataset(self, partition: str) -> Dataset:
        pass

    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def build_dataloader(self, partition: str) -> DataLoader:
        pass

    @abstractmethod
    def build_optimizer_scheduler(self, model: torch.nn.Module) -> Tuple[Optimizer, _LRScheduler]:
        pass

    @abstractmethod
    def save_checkpoint(self) -> None:
        pass

    @abstractmethod
    def load_checkpoint(self) -> None:
        pass

    @abstractmethod
    def build_loss_function(self) -> Any:
        pass

    @abstractmethod
    def train_epoch_start(self) -> None:
        pass

    # TODO: Correct the type hint for loss function
    @abstractmethod
    def train_step(
        self,
        model: torch.nn.Module,
        data_batch: DataBatch,
        loss_function: torch.nn.Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler
        ) -> None:
        pass

    @abstractmethod
    def train_epoch_end(self) -> None:
        pass

    @abstractmethod
    def val_epoch_start(self) -> None:
        pass

    # TODO: Correct the type hint for loss function and criterion
    @abstractmethod
    def val_step(
        self,
        model: torch.nn.Module,
        data_batch: DataBatch,
        loss_function: torch.nn.Module,
        criterion: torch.nn.Module
        ) -> None:
        pass
    
    @abstractmethod
    def val_epoch_end(self) -> None:
        pass
