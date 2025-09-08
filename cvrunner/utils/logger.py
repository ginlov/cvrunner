import logging
import sys
import os


try:
    import wandb
except ImportError:
    wandb = None

import cvrunner.utils.distributed as dist

from datetime import datetime

class CVLogger(logging.Logger):
    """
    General-purpose logger for CVRunner.
    Reads all Weights & Biases configuration from environment variables.

    Env vars:
      - WANDB_PROJECT    (project name, required to enable W&B)
      - WANDB_RUN_NAME   (run name, optional)
      - WANDB_BASE_URL   (self-hosted W&B URL, optional; defaults to official cloud)
      - WANDB_API_KEY    (API key for authentication, required if not already logged in)
    """

    def __init__(self, name="cvrunner", level=logging.INFO, wandb_project: None | str = None, wandb_runname: None | str = None):
        super().__init__("cvrunner", level)

        # Console logging only once (on rank 0)
        if not self.handlers and dist.is_main_process():
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.addHandler(handler)

        # W&B configuration from environment
        wandb_url = os.getenv("WANDB_BASE_URL", "https://api.wandb.ai")
        api_key = os.getenv("WANDB_API_KEY")

        self._wandb_enabled = False
        if wandb is not None and wandb_project is not None:
            if dist.is_main_process() and wandb.run is None:
                os.environ["WANDB_BASE_URL"] = wandb_url
                if api_key:
                    os.environ["WANDB_API_KEY"] = api_key  # ensures login works
                try:
                    wandb.login(key=api_key) if api_key else wandb.login()
                except Exception as e:
                    self.warning(f"Failed to login to W&B: {e}")
                # If runname is None, use current time to create run name
                if wandb_runname is None:
                    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    wandb_runname = f"{wandb_project}-{timestamp}"
                try:
                    wandb.init(project=wandb_project, name=wandb_runname)
                    self._wandb_enabled = True
                except Exception as e:
                    self.warning(f"Failed to init W&B: {e}")
        else:
            if dist.is_main_process():
                self.info("W&B logging disabled (set WANDB_PROJECT to enable).")

    def log_metrics(self, metrics: dict, local_step: int):
        """
        Log metrics with global step aligned across all ranks.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        global_step = local_step * world_size + rank

        # Console (only rank 0)
        if dist.is_main_process():
            msg = " | ".join(
                [
                    f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                    for k, v in metrics.items()
                ]
            )
            self.info(f"[global_step {global_step}] {msg}")

        # W&B (all ranks can contribute)
        if self._wandb_enabled and wandb.run is not None:
            wandb.log(metrics, step=global_step)


# Singleton
_logger = None


def get_cv_logger(level=logging.INFO, wandb_project: None | str = None, wandb_runname: None | str = None):
    """
    Get a singleton logger instance.
    Uses console logging always, W&B logging only if env vars are set.
    """
    global _logger
    if _logger is None:
        logging.setLoggerClass(CVLogger)
        _logger = logging.getLogger("cvrunner")
        _logger.__init__(level=level, wandb_project=wandb_project, wandb_runname=wandb_runname)
    return _logger
