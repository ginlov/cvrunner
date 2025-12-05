import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from cvrunner.runner.train_runner import TrainRunner
from cvrunner.experiment.experiment import BaseExperiment
from cvrunner.utils.logger import get_cv_logger

logger = get_cv_logger()

class DistributedTrainRunner(TrainRunner):
    def __init__(
        self,
        experiment: BaseExperiment
    ) -> None:
        super().__init__(experiment)
