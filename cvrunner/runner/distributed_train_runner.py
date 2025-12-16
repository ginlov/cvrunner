import sys
import pathlib
import torch
import torch.distributed as dist
import warnings
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Add project root to path for module execution
sys.path.insert(0, str(pathlib.Path.cwd()))

from cvrunner.runner.train_runner import TrainRunner
from cvrunner.utils.logger import get_cv_logger
from cvrunner.utils.distributed import setup_distributed, cleanup_distributed, is_main_process

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore", module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="wandb.analytics.sentry")


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
        self.local_rank, self.dist_device = setup_distributed()
        
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

            # Validation
            if epoch % val_freq == 0:
                # Barrier ensures all processes finish training before validation starts
                dist.barrier()
                
                logger.info(f"Start validation at epoch {epoch}.")
                self.val_epoch_start()
                self.val_epoch()
                self.val_epoch_end()
                logger.info(f"Done validation at epoch {epoch}.")
                
                # Checkpointing (Only Rank 0)
                if is_main_process():
                    self.checkpoint()

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

# --- ENTRY POINT ---
if __name__ == "__main__":
    import argparse
    import importlib.util
    from cvrunner.experiment.experiment import BaseExperiment
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", type=str, required=True)
    args = parser.parse_args()

    # 1. Load Experiment Class Dynamically
    exp_path = pathlib.Path(args.experiment_path).resolve()
    spec = importlib.util.spec_from_file_location("dynamic_exp", exp_path)
    
    # Assertion whether spec and loader are valid
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load experiment module from {exp_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    ExperimentClass = None
    for attr in dir(module):
        obj = getattr(module, attr)
        if isinstance(obj, type) and issubclass(obj, BaseExperiment) and obj is not BaseExperiment:
            ExperimentClass = obj
            break
            
    if not ExperimentClass:
        raise RuntimeError("No BaseExperiment subclass found.")
    
    # 2. Instantiate Experiment
    experiment = ExperimentClass()
    
    # 3. Initialize W&B (Guarded by Rank 0 inside logger)
    if experiment.wandb_project:
        # Assuming you have a helper to extract config properties
        logger.init_wandb(experiment.wandb_project, experiment.wandb_runname, config=vars(experiment))

    # 4. Run Distributed Training
    runner_cls = experiment.runner_cls()
    runner = runner_cls(experiment)
    runner.run()
    
    # 5. Cleanup
    cleanup_distributed()
