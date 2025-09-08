from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Tuple, Union, Dict, Type, TYPE_CHECKING

import torch

from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from cvrunner.utils.logger import get_cv_logger

logger = get_cv_logger()

if TYPE_CHECKING:
    from cvrunner.runner.runner import BaseRunner

DataBatch = Dict[str, Union[Any, torch.tensor]]

class BaseExperiment(ABC):
    """
    Template for the configuration.
    Every experiments should be stateless.
    """
    @property
    @abstractmethod
    def runner_cls(self) -> Type[BaseRunner]:
        """
        Define type of runner to run this experiment

        Returns:
            Type[BaseRunner]: runner class
        """
        pass
    
    @property
    def batch_size(self) -> int:
        """
        Returns:
            int: batch size
        """
        return 0
    
    @property
    def num_epochs(self) -> int:
        """
        Returns:
            int: num epochs
        """
        return 0
    
    @property
    def val_freq(self) -> int:
        """
        Returns:
            int: validation frequency
        """
        return 10
    
    @property
    def wandb_project(self) -> str:
        return 'default-cvrunner'
    
    @property
    def wanbd_runname(self) -> None | str:
        return None
    
    @abstractmethod
    def __init__(self) -> None:
        """
        Init neccessary things for experiment.
        Everything should be stateless.
        """
        pass

    @abstractmethod
    def build_dataset(self, partition: str) -> Dataset:
        """
        Dataset building logic

        Args:
            partition (str): should be in ['train', 'val']

        Returns:
            Dataset:
        """
        pass

    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        """
        Model building logic

        Returns:
            torch.nn.Module:
        """
        pass

    @abstractmethod
    def build_dataloader(self, partition: str) -> DataLoader:
        """
        Dataloader building logic

        Args:
            partition (str): should be in ['train', 'val']

        Returns:
            DataLoader:
        """
        pass

    @abstractmethod
    def build_optimizer_scheduler(self, model: torch.nn.Module) -> Tuple[Optimizer, _LRScheduler]:
        """
        Optimizer and Learning scheduler building logic

        Args:
            model (torch.nn.Module): model to be trained

        Returns:
            Tuple[Optimizer, _LRScheduler]:
        """
        pass

    @abstractmethod
    def save_checkpoint(self) -> None:
        """
        Checkpointing logic
        """
        pass

    @abstractmethod
    def load_checkpoint(self, file_path: str) -> torch.nn.Module:
        """_summary_

        Args:
            file_path (str): checkpoint point file

        Returns:
            torch.nn.Module:
        """
        pass

    @abstractmethod
    def build_loss_function(self) -> Any:
        """
        Loss function building logic

        Returns:
            Any: callable loss function
        """

    def train_epoch_start(self) -> None:
        """
        Train epoch startiting methods
        """
        logger.info("Nothing happens before train epoch starts")

    # TODO: Correct the type hint for loss function
    def train_step(
        self,
        model: torch.nn.Module,
        data_batch: DataBatch,
        loss_function: torch.nn.Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler
        ) -> None:
        """
        Train step logic

        Args:
            model (torch.nn.Module): model to be trained
            data_batch (DataBatch): data to train
            loss_function (torch.nn.Module): loss function
            optimizer (Optimizer):
            lr_scheduler (_LRScheduler):
        """
        logger.info("Empty training step")

    def train_epoch_end(self) -> None:
        """
        Train epoch ending methods
        """
        logger.info("Nothing happens after train epoch ends")

    def val_epoch_start(self) -> None:
        """
        Validation epoch startining methods
        """
        logger.info("Nothing happens before val epoch starts")

    # TODO: Correct the type hint for loss function and criterion
    def val_step(
        self,
        model: torch.nn.Module,
        data_batch: DataBatch,
        loss_function: torch.nn.Module,
        criterion: torch.nn.Module
        ) -> None:
        """
        validation step logic

        Args:
            model (torch.nn.Module):
            data_batch (DataBatch):
            loss_function (torch.nn.Module):
            criterion (torch.nn.Module):
        """
        logger.info("Empty validation step")
    
    def val_epoch_end(self) -> None:
        """
        Validation epoch ending logic
        """
        logger.info("Nothing happens after val epoch ends")
