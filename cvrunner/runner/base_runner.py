import random
import numpy as np
import torch

from abc import ABC, abstractmethod
from typing import Any, Type

from cvrunner.experiment.experiment import BaseExperiment
from cvrunner.utils.logger import get_cv_logger

logger = get_cv_logger()

class BaseRunner(ABC):
    """Abstract class for runners

    Args:
        ABC (_type_): _description_
    """
    @abstractmethod
    def __init__(
        self,
        experiment: Type[BaseExperiment]
        ) -> None:
        """
        Args:
            experiment (Type[BaseExperiment]): experiment to run
        """
        pass

    def _fix_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    @abstractmethod
    def run(self):
        """
        To run experiment
        """
        pass

    def checkpoint(self):
        """
        For experiment checkpointing
        """
        self.experiment.save_checkpoint()

    def train_epoch_start(self):
        """
        For train epoch starting methods
        """
        self.experiment.train_epoch_start()

    @abstractmethod
    def train_epoch(self):
        """
        For train epoch logic
        """
        pass

    def train_epoch_end(self):
        """
        For train epoch ending methods
        """
        self.experiment.train_epoch_end()

    def val_epoch_start(self):
        """
        For val epoch starting methods
        """
        self.experiment.val_epoch_start()
    
    @abstractmethod
    def val_epoch(self):
        """
        For val epoch logic
        """
        pass

    def val_epoch_end(self):
        """
        For val epoch ending methods
        """
        self.experiment.val_epoch_end()
    
    @abstractmethod
    def train_step(self, data: Any):
        """
        Train step logic

        Args:
            data (Any): input data to train
        """
        pass

    @abstractmethod
    def val_step(self, data: Any):
        """
        Validation step logic

        Args:
            data (Any): input data to validate
        """
        pass