import argparse
import importlib
import sys
import pathlib

from typing import Type

from cvrunner.runner.runner import BaseRunner
from cvrunner.experiment.experiment import BaseExperiment

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

    # Find subclass of BaseExperiment
    exp_classes = [
        obj for obj in module.__dict__.values()
        if isinstance(obj, type) and issubclass(obj, BaseExperiment) and obj is not BaseExperiment
    ]
    if len(exp_classes) != 1:
        raise RuntimeError(
            f"Expected exactly 1 Experiment subclass in {exp_path}, found {len(exp_classes)}"
        )
    return exp_classes[0]

def main():
    """
    Entry point for cvrunner
    """
    parser = argparse.ArgumentParser(description="Run experiments with cvrunner.")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Path to experiment Python file (must contain exactly one Experiment class)."
    )
    args = parser.parse_args()

    ExpClass = load_experiment_class(args.exp)
    exp: BaseExperiment = ExpClass()
    runner_cls = exp.runner_cls
    runner: BaseRunner = runner_cls(exp)
    runner.run()
