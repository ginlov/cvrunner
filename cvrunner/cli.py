import argparse
import importlib
import sys
import pathlib
import subprocess
import os

from typing import Type
from argparse import Namespace

from cvrunner.runner.base_runner import BaseRunner
from cvrunner.experiment.experiment import BaseExperiment
from cvrunner.utils.logger import get_cv_logger

# Add current working directory to sys.path
sys.path.insert(0, str(pathlib.Path.cwd()))

def load_experiment_class(exp_path: str) -> Type[BaseExperiment]:
    """Load experiment class from experiment file

    Args:
        exp_path (str): path to exp file

    Raises:
        FileNotFoundError: if exp_path file does not exist
        RuntimeError: exp_path file has more than 1 experiment class

    Returns:
        Type[BaseExperiment]: Experiment class
    """
    exp_path = pathlib.Path(exp_path).resolve()
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {exp_path}")

    # Dynamically load module
    spec = importlib.util.spec_from_file_location("experiment_module", exp_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["experiment_module"] = module
    spec.loader.exec_module(module)

    # Find subclass of BaseExperiment defined in *this* module
    exp_classes = [
        obj for obj in module.__dict__.values()
        if (
            isinstance(obj, type) 
            and issubclass(obj, BaseExperiment) 
            and obj is not BaseExperiment
            and obj.__module__ == module.__name__  # only classes defined here
        )
    ]
    if len(exp_classes) != 1:
        raise RuntimeError(
            f"Expected exactly 1 Experiment subclass in {exp_path}, found {len(exp_classes)}"
        )
    return exp_classes[0]

def run_local(args: Namespace):
    ExpClass = load_experiment_class(args.exp)
    exp: BaseExperiment = ExpClass()

    runner_cls = exp.runner_cls()
    wandb_project = exp.wandb_project
    wandb_runname = exp.wandb_runname

    logger = get_cv_logger()
    if wandb_project is not None:
        logger.init_wandb(project=wandb_project, run_name=wandb_runname, config=vars(exp))

    logger.info("Successfully initialized experiment.")
    logger.info("Start initializing runner")

    runner: BaseRunner = runner_cls(exp)
    logger.info("Start running runner")
    runner.run()

def docker_image_exists(image: str) -> bool:
    """Check if a docker image exists locally."""
    result = subprocess.run(
        ["docker", "images", "-q", image],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.stdout.strip() != ""

def build_docker_image(image: str):
    """Build docker image using docker-compose in environments/."""
    env_dir = pathlib.Path(__file__).resolve().parent.parent / "environments"
    print(f"[CVRUNNER] Building docker image {image} from {env_dir}...")
    subprocess.run(["docker-compose", "-f", str(env_dir / "docker-compose.yml"), "build"], check=True)

def run_in_docker(exp_path, extra_args, build):
    """Launch Docker container and run training inside"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    image_name = "cvrunner:latest"

    # Auto-build if needed
    if build or not docker_image_exists(image_name):
        build_docker_image(image_name)

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{project_root}:/workspace",
        "-w", "/workspace",
        "cvrunner:latest",  # Docker image name
        "-l",
        "--exp", exp_path
    ] + extra_args

    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run experiments with cvrunner.")
    parser.add_argument("-e", "--exp", type=str, required=True)
    parser.add_argument("-l", "--run-local", action="store_true", help="Run the experiment locally instead of inside Docker")
    parser.add_argument("--build", action="store_true", help="Build Docker image before running")
    args = parser.parse_args()

    if args.run_local:
        run_local(args)
    else:
        exp_path = args.exp
        exp_args = []
        run_in_docker(exp_path, exp_args, args.build)