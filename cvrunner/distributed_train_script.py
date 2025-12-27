import argparse
import importlib.util
from cvrunner.experiment.experiment import BaseExperiment
from cvrunner.utils.utils import get_properties
import pathlib
from cvrunner.utils.distributed import setup_distributed, cleanup_distributed
from cvrunner.utils.logger import get_cv_logger, reconfigure_cv_logger
# Ignore warnings for cleaner output
import warnings
# Add current working directory to sys.path

warnings.filterwarnings("ignore", module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="wandb.analytics.sentry")

logger = get_cv_logger()

# --- ENTRY POINT ---
if __name__ == "__main__":
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
    
    # 3. Initialize distributed and  W&B (Guarded by Rank 0 inside logger)
    _, __ = setup_distributed()
    reconfigure_cv_logger()
    if experiment.wandb_project:
        # Assuming you have a helper to extract config properties
        logger.info(f"Config: {vars(experiment)}")
        logger.init_wandb(experiment.wandb_project, experiment.wandb_runname, config=get_properties(experiment))

    # 4. Run Distributed Training
    runner_cls = experiment.runner_cls()
    runner = runner_cls(experiment)
    runner.run()
    
    # 5. Cleanup
    cleanup_distributed()
