import logging
import wandb
import sys

import cvrunner.utils.distributed as dist

class DetrLogger(logging.Logger):
    def __init__(self, name="detr", level=logging.INFO, project="torch-detr",
                 run_name=None, config=None, use_wandb=True):
        super().__init__(name, level)

        # Console logging only once
        if not self.handlers and dist.is_main_process():
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.addHandler(handler)

        # Init W&B only on rank 0 (others attach automatically)
        if use_wandb and wandb.run is None and dist.is_main_process():
            wandb.init(project=project, name=run_name, config=config)

    def log_metrics(self, metrics: dict, local_step: int):
        """
        Log metrics with global step aligned across all ranks.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Compute global step (interleaved)
        global_step = local_step * world_size + rank

        # Console only on main process
        if dist.is_main_process():
            msg = " | ".join(
                [f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                 for k, v in metrics.items()]
            )
            self.info(f"[global_step {global_step}] {msg}")

        # W&B: all ranks log
        if wandb.run is not None:
            wandb.log(metrics, step=global_step)


# Singleton
_detr_logger = None

def get_detr_logger(project="torch-detr", run_name=None, config=None,
                    level=logging.INFO, use_wandb=True):
    global _detr_logger
    if _detr_logger is None:
        logging.setLoggerClass(DetrLogger)
        _detr_logger = logging.getLogger("detr")
        _detr_logger.__init__("detr", level, project, run_name, config, use_wandb)
    return _detr_logger
