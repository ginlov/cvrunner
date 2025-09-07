from abc import ABC, abstractmethod
from typing import Any

from cvrunner.experiment.experiment import BaseExperiment

class BaseRunner(ABC):
    @abstractmethod
    def __init__(
        self,
        experiment: BaseExperiment
        ) -> None:
        pass
    
    @abstractmethod
    def run(self):
        pass

    def checkpoint(self):
        pass

    def train_epoch_start(self):
        pass

    def train_epoch(self):
        pass

    def train_epoch_end(self):
        pass

    def val_epoch_start(self):
        pass

    def val_epoch(self):
        pass

    def val_epoch_end(self):
        pass

    def train_step(self):
        pass

    def val_step(self):
        pass

class TrainRunner(BaseRunner):
    def __init__(
            self,
            experiment: BaseExperiment
    ) -> None:
        self.experiment = experiment
        
        # build model
        self.model = self.experiment.build_model()
        
        # build loss function
        self.loss_function = self.experiment.build_loss_function()

        # build data loader
        self.train_dataloader = self.experiment.build_dataloader(partition='train')
        self.val_dataloader = self.experiment.build_dataloader(partition='val')

        # build optimizer
        self.optimizer, self.lr_scheduler = self.experiment.build_optimizer_scheduler(self.model)

    def run(self):
        num_epoch = self.experiment.num_epochs
        val_freq = self.experiment.val_freq
        for epoch in range(num_epoch):
            self.train_epoch_start()
            self.train_epoch()
            self.train_epoch_end()

            if epoch % val_freq == 0:
                self.val_epoch_start()
                self.val_epoch()
                self.val_epoch_end()

    def train_epoch(self):
        for data in self.train_dataloader:
            self.train_step(data)

    def val_epoch(self):
        for data in self.val_dataloader:
            self.val_step(data)

    def train_step(self, data: Any):
        self.experiment.train_step(
            self.model,
            data,
            self.loss_function,
            self.optimizer,
            self.lr_scheduler
        )

    def val_step(self, data: Any):
        self.experiment.val_step(
            self.model,
            data,
            self.loss_function,
            None
        )