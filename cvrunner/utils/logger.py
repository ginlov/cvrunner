import logging
import sys
import os
import wandb
import numpy as np
from typing import List, Union
from datetime import datetime
from PIL import Image
from cvrunner.utils.distributed import get_rank

import cvrunner.utils.distributed as dist

class CVLogger(logging.Logger):
    def __init__(self, name="cvrunner", level=logging.INFO):
        super().__init__(name, level)
        self.propagate = False

        if not self.handlers:
            handler = logging.StreamHandler(sys.__stdout__)
            handler.setLevel(level)
            formatter = logging.Formatter(
                f"[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.addHandler(handler)

        self._wandb_enabled = False
        self._wandb_runname = None

    def init_wandb(self, project: str, run_name: str | None = None, config=None):
        """
        Initialize W&B logging. Only runs on Rank 0.
        """
        # Guard: Only Rank 0 can initialize W&B
        if not dist.is_main_process():
            return

        if run_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            run_name = f"{project}-{timestamp}"

        # Load API keys
        wandb_url = os.getenv("WANDB_BASE_URL", "https://api.wandb.ai")
        api_key = os.getenv("WANDB_API_KEY")
        
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
            wandb.login(key=api_key, host=wandb_url)

        try:
            wandb.init(project=project, name=run_name, config=config)
            self._wandb_enabled = True
            self._wandb_runname = run_name
            self.info(f"🚀 W&B Initialized: {run_name}")
        except Exception as e:
            self.warning(f"Failed to init W&B: {e}")

    def log_metrics(self, metrics: dict, step: int, average_across_ranks=True):
        """
        Log metrics (scalars).
        
        Args:
            metrics: Dict of values (e.g. {'loss': 0.5, 'acc': 0.9})
            step: Global step (usually epoch or iteration)
            average_across_ranks: If True, averages values from all GPUs (Recommended).
        """
        # 1. Synchronize/Reduce metrics
        if average_across_ranks:
            # Averages loss across all GPUs so you get one clean line
            try:
                metrics = dist.reduce_dict(metrics)
            except Exception as e:
                print(metrics)
                raise e
        
        # 2. Log only on Rank 0
        if dist.is_main_process():
            # Console log
            msg = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.info(f"[Step {step}] {msg}")

            # W&B log
            if self._wandb_enabled:
                wandb.log(metrics, step=step)

    def log_images(self, image_ids: List[int], images: List[Union[np.ndarray, Image.Image]], step: int):
        """
        Log images to W&B (Only logs images present on Rank 0).
        """
        if not self._wandb_enabled or not dist.is_main_process():
            return

        wandb_images = []
        for img_id, img in zip(image_ids, images):
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif isinstance(img, str):
                img = Image.open(img)
            wandb_images.append(wandb.Image(img, caption=f"{img_id}"))

        wandb.log({
            str(img_id): wandb.Image(img) \
            for img_id, img in zip(image_ids, images)
        }, step=step)

# Singleton Pattern
_logger = None

def get_cv_logger(level=logging.INFO):
    global _logger
    if _logger is None:
        rank = get_rank()
        logging.setLoggerClass(CVLogger)
        logger = logging.getLogger("cvrunner")
        logger.setLevel(level)
        _logger = logger
    return _logger

def reconfigure_cv_logger():
    """Call this after torch.distributed is initialized to update the logger's rank in the format."""
    global _logger
    if _logger is not None:
        rank = get_rank()
        for handler in _logger.handlers:
            handler.setFormatter(logging.Formatter(f"[Rank {rank}] [%(levelname)s] %(message)s"))

