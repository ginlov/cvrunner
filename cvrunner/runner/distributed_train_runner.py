import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from cvrunner.runner.train_runner import TrainRunner
from cvrunner.utils.logger import get_cv_logger
from cvrunner.utils.distributed import is_main_process

logger = get_cv_logger()

class DistributedTrainRunner(TrainRunner):
    """
    Runner for Distributed Data Parallel (DDP) training.
    Inherits standard logic from TrainRunner but overrides:
      1. Device placement (Rank specific)
      2. Model wrapping (DDP + SyncBatchNorm)
      3. Sampler shuffling (set_epoch)
      4. Logging aggregation
    """

    def __init__(self, experiment):
        # 1. Initialize Distributed Group & Get Local Rank
        # We do this *before* super().__init__ because we might need rank info,
        # but TrainRunner initializes components immediately.
        self.local_rank = dist.get_rank()
        self.dist_device = torch.device(f"cuda:{self.local_rank}")
        
        # 2. Initialize Parent (Builds model, optimizer, etc.)
        super().__init__(experiment)

        # 3. Override Device
        # TrainRunner defaults to "cuda", but we need "cuda:N"
        self.device = self.dist_device
        
        logger.info(f"[Rank {self.local_rank}] Moving model to {self.device}...")
        self.model = self.model.to(self.device)

        # 4. Sync Batch Norm
        # Essential for CV: synchronizes stats across GPUs for small per-GPU batches
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # 5. Wrap in DDP
        self.model = DDP(
            self.model, 
            device_ids=[self.local_rank], 
            output_device=self.local_rank,
            # find_unused_parameters=True # Uncomment if you have dynamic control flow
        )
        logger.info(f"[Rank {self.local_rank}] DDP Model Initialized")

    def run(self):
        """
        DDP-specific run loop. 
        Main difference: We MUST call set_epoch on the DistributedSampler.
        """
        num_epoch = self.experiment.num_epochs
        val_freq = self.experiment.val_freq

        logger.info(f"Start Distributed Training: {num_epoch} epochs")

        for epoch in range(num_epoch):
            # --- CRITICAL DDP STEP ---
            # Ensures each GPU gets a different slice of data every epoch
            if hasattr(self.train_dataloader, 'sampler') and \
               isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            logger.info(f"Start training epoch {epoch}.")
            
            self.train_epoch_start()
            self.train_epoch()
            self.train_epoch_end()
            
            logger.info(f"Done training epoch {epoch}")

            dist.barrier()
            # Validation
            if epoch % val_freq == 0 and is_main_process():
                logger.info(f"Start validation at epoch {epoch}.")
                self.val_epoch_start()
                self.val_epoch()
                self.val_epoch_end()
                logger.info(f"Done validation at epoch {epoch}.")
                
                # Checkpointing (Only Rank 0)
                self.checkpoint()

            dist.barrier()

    def train_step(self, data):
        """
        Override to ensure data moves to the specific LOCAL RANK device.
        """
        data = self._move_to_device(data)
        
        # Pass to experiment
        metrics = self.experiment.train_step(
            self.model,
            data,
            self.loss_function,
            self.optimizer,
            self.lr_scheduler,
            self.device
        )
        
        # Log Metrics
        # 'average_across_ranks=True' ensures W&B gets the smooth average loss
        logger.log_metrics(metrics, step=self.step, average_across_ranks=True)
        self.step += 1

    def val_step(self, data):
        """
        Override to ensure data moves to the specific LOCAL RANK device.
        """
        data = self._move_to_device(data)
        
        metrics = self.experiment.val_step(
            self.model,
            data,
            self.loss_function,
            self.criterion,
            self.device
        )
        # Update local aggregator
        self.val_metrics.update(metrics)

    def val_epoch_end(self):
        # When logging summary, we want to reduce (average) across all GPUs
        # Note: logger.log_metrics handles reduction internally via `dist.reduce_dict`
        logger.log_metrics(
            self.val_metrics.summary(), 
            step=self.step, 
            average_across_ranks=True
        )
        self.val_metrics.reset()

    def _move_to_device(self, data):
        """Recursive helper to move data to the local rank device."""
        if torch.is_tensor(data):
            return data.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._move_to_device(v) for v in data]
        elif isinstance(data, tuple):
            return tuple(self._move_to_device(v) for v in data)
        return data


