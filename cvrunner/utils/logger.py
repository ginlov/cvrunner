import logging
import sys
import os
import wandb
import numpy as np
from typing import List, Union
from datetime import datetime
from PIL import Image

import cvrunner.utils.distributed as dist


class CVLogger(logging.Logger):
    """
    General-purpose logger for CVRunner.
    Console logging is always enabled.
    W&B logging must be explicitly initialized via `init_wandb`.
    """

    def __init__(self, name="cvrunner", level=logging.INFO):
        super().__init__(name, level)
        self.propagate = False

        # Console logging only once (on rank 0)
        if not self.handlers and dist.is_main_process():
            handler = logging.StreamHandler(sys.__stdout__)  # safe against pytest
            handler.setLevel(level)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.addHandler(handler)

        self._wandb_enabled = False
        self._wandb_project = None
        self._wandb_runname = None

    def init_wandb(self, project: str, run_name: str | None = None, config=None):
        """
        Initialize W&B logging. Should be called once from the CLI entrypoint.
        Reads additional env vars:
          - WANDB_BASE_URL
          - WANDB_API_KEY
        """
        self._wandb_project = project

        # Generate default run name if not given
        if run_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            run_name = f"{project}-{timestamp}"
        self._wandb_runname = run_name

        wandb_url = os.getenv("WANDB_BASE_URL", "https://api.wandb.ai")
        api_key = os.getenv("WANDB_API_KEY")

        if dist.is_main_process():
            os.environ["WANDB_BASE_URL"] = wandb_url
            if api_key:
                os.environ["WANDB_API_KEY"] = api_key
                try:
                    wandb.login(key=api_key)
                except Exception as e:
                    self.warning(f"Failed to login to W&B: {e}")
            try:
                wandb.init(project=project, name=run_name, config=config)
                self._wandb_enabled = True
                self.info(f"W&B initialized: project={project}, run={run_name}")
            except Exception as e:
                self.warning(f"Failed to init W&B: {e}")

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

        # W&B (if enabled)
        if self._wandb_enabled and wandb.run is not None:
            wandb.log(metrics, step=global_step)

    def log_images(
            self,
            image_ids: List[int],
            images: List[Union[np.ndarray, Image.Image]],
            local_step: int
    ):
        """
        Log images to W&B with image IDs and time step.

        Args:
            image_ids (List[int]): Unique identifiers for each image.
            images (List[Union[np.ndarray, Image.Image]]): List of images.
            local_step (int): The time/global step for logging.
        """
        if not self._wandb_enabled or wandb.run is None:
            self.warning("W&B not initialized. Skipping image logging.")
            return

        if not (isinstance(image_ids, list) and isinstance(images, list)):
            self.warning("image_ids and images must be lists.")
            return

        if len(image_ids) != len(images):
            self.warning("Number of image_ids must match number of images.")
            return

        wandb_images = []
        for img_id, img in zip(image_ids, images):
            try:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                elif isinstance(img, str):
                    img = Image.open(img)
                # Optionally, add more validation here
                wandb_images.append(wandb.Image(img, caption=f"{img_id}"))
            except Exception as e:
                self.warning(f"Failed to process image {img_id}: {e}")

        if wandb_images:
            wandb.log({
                str(img_id): wandb.Image(img) for img, img_id in zip(wandb_images, image_ids)
            }, step=local_step)

    def log_histogram(self, name: str, values: np.ndarray, local_step: int):
        """
        Log a histogram of values to W&B.

        Args:
            name (str): Name of the histogram.
            values (np.ndarray): Values to create the histogram from.
            local_step (int): The time/global step for logging.
        """
        if not self._wandb_enabled or wandb.run is None:
            self.warning("W&B not initialized. Skipping histogram logging.")
            return

        if not isinstance(values, np.ndarray):
            self.warning("values must be a numpy array.")
            return
        
        self.info(f"Logging histogram '{name}' at step {local_step} with {len(values)} values.")
        wandb.log({name: wandb.Histogram(values)}, step=local_step)

# Singleton
_logger = None

def get_cv_logger(level=logging.INFO) -> CVLogger:
    """
    Get a singleton logger instance.
    Console logging is always enabled.
    W&B logging must be explicitly initialized later via logger.init_wandb().
    """
    global _logger
    if _logger is None:
        logging.setLoggerClass(CVLogger)
        _logger = logging.getLogger("cvrunner")
        _logger.__init__(level=level)  # re-init as CVLogger
    return _logger
