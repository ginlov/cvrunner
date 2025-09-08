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
        self.experiment = experiment
    
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
        logger.info("Start checkpointing")
        self.experiment.save_checkpoint()
        logger.info("Done checkpointing")

    def train_epoch_start(self):
        """
        For train epoch starting methods
        """
        self.experiment.train_epoch_start()

    def train_epoch(self):
        """
        For train epoch logic
        """
        logger.info("Empty training epoch")

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

    def val_epoch(self):
        """
        For val epoch logic
        """
        logger.info("Empty validation epoch")

    def val_epoch_end(self):
        """
        For val epoch ending methods
        """
        self.experiment.val_epoch_end()

    def train_step(self, data: Any):
        """
        Train step logic

        Args:
            data (Any): input data to train
        """
        logger.info("Empty train step")

    def val_step(self, data: Any):
        """
        Validation step logic

        Args:
            data (Any): input data to validate
        """
        logger.info("Empty validation step")

class TrainRunner(BaseRunner):
    """
    Simple Runner for training
    """
    def __init__(
            self,
            experiment: BaseExperiment
    ) -> None:
        """
        Init runner, build model, loss functions, metrics and related components

        Args:
            experiment (BaseExperiment): _description_
        """
        logger.info("START INITIALIZING TRAIN RUNNER")
        self.experiment = experiment
        
        logger.info(f"Building model")
        # build model
        self.model = self.experiment.build_model()
        logger.info("Done building model")
        
        logger.info("Building loss function")
        # build loss function
        self.loss_function = self.experiment.build_loss_function()
        logger.info("Done building loss function")

        logger.info("Building dataloaders")
        # build data loader
        self.train_dataloader = self.experiment.build_dataloader(partition='train')
        self.val_dataloader = self.experiment.build_dataloader(partition='val')
        logger.info("Done building dataloaders")

        logger.info("Building optimizer and learning rate scheduler")
        # build optimizer
        self.optimizer, self.lr_scheduler = self.experiment.build_optimizer_scheduler(self.model)
        logger.info("Done building optimizer and learning rate scheduler")

        logger.info("DONE INITIALIZING TRAIN RUNNER")

    def run(self):
        """
        To run experiment. General logic is

        for epoch in range(num_epochs):
        
            train_epoch_start()
            
            train_epoch()
            
            train_epoch_end()

            if val_epoch:
            
                val_epoch_start()
                
                val_epoch()
                
                val_epoch_end()
        """
        num_epoch = self.experiment.num_epochs
        val_freq = self.experiment.val_freq
        logger.info(f"Start training model with {num_epoch} epochs and validating model every {val_freq} epochs.")
        for epoch in range(num_epoch):
            logger.info(f"Start training epoch {epoch}.")
            self.train_epoch_start()
            self.train_epoch()
            self.train_epoch_end()
            logger.info(f"Done training epoch {epoch}")

            if epoch % val_freq == 0:
                logger.info(f"Start validation at epoch {epoch}.")
                self.val_epoch_start()
                self.val_epoch()
                self.val_epoch_end()
                logger.info(f"Done validaiton at epoch {epoch}.")

    def train_epoch(self):
        """
        Train epoch logic
        """
        for data in self.train_dataloader:
            self.train_step(data)

    def val_epoch(self):
        """
        Validation epoch logic
        """
        for data in self.val_dataloader:
            self.val_step(data)

    def train_step(self, data: Any):
        """
        Train step logic, basically call train_step function of experiment.

        Args:
            data (Any): _description_
        """
        self.experiment.train_step(
            self.model,
            data,
            self.loss_function,
            self.optimizer,
            self.lr_scheduler
        )

    def val_step(self, data: Any):
        """
        Validation step logic, basically call val_step function of experiment.

        Args:
            data (Any): _description_
        """
        self.experiment.val_step(
            self.model,
            data,
            self.loss_function,
            None
        )