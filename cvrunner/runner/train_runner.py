import torch

from typing import Any
from torch.nn.parallel import DistributedDataParallel as DDP

from cvrunner.experiment.experiment import BaseExperiment
from cvrunner.runner.base_runner import BaseRunner
from cvrunner.utils.logger import get_cv_logger
from cvrunner.utils.utils import MetricAggregator, MultiMetricAggregator

logger = get_cv_logger()

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
        self._fix_seed(self.experiment.random_seed)

        # detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        logger.info(f"Building model")
        # build model
        self.model = self.experiment.build_model().to(self.device)
        # self.model = DDP(self.model, device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None)
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

        # Initial step for logging
        self.step = 0

        # Initial validation metrics aggregator
        self.valid_metrics = MultiMetricAggregator()

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
        metrics = self.experiment.train_step(
            self.model,
            data,
            self.loss_function,
            self.optimizer,
            self.lr_scheduler,
            self.device
        )
        logger.log_metrics(metrics, local_step=self.step)
        self.step += 1

    def val_epoch_end(self):
        super().val_epoch_end()

        logger.log_metrics(self.valid_metrics.summary(), local_step=self.step)
        self.valid_metrics.reset()
    
    # TODO: Create MetricAggregator to aggregate validation metrics and logging
    def val_step(self, data: Any):
        """
        Validation step logic, basically call val_step function of experiment.

        Args:
            data (Any): _description_
        """
        metrics = self.experiment.val_step(
            self.model,
            data,
            self.loss_function,
            None,
            self.device
        )
        self.valid_metrics.update(metrics)