import argparse
import importlib
import sys
import pathlib
import subprocess
import os
import tempfile
import yaml
import textwrap

from typing import Type, List
from argparse import Namespace

from cvrunner.runner.base_runner import BaseRunner
from cvrunner.experiment.experiment import BaseExperiment
from cvrunner.utils.logger import get_cv_logger

# Add current working directory to sys.path
sys.path.insert(0, str(pathlib.Path.cwd()))
logger = get_cv_logger()

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

def run_local(args: Namespace) -> None:
    """Run experiment locally.

    Args:
        args (Namespace): argurments for training job    """
    ExpClass = load_experiment_class(args.exp)
    exp: BaseExperiment = ExpClass()

    runner_cls = exp.runner_cls()
    wandb_project = exp.wandb_project
    job_name = os.environ.get("JOB_NAME")
    # TODO: a bit tricky here, clean it later
    wandb_runname = "-".join(job_name.split('-')[:-1]) if job_name else exp.wandb_runname  # fallback for local runs

    if wandb_project is not None:
        logger.init_wandb(project=wandb_project, run_name=wandb_runname, config=vars(exp))

    logger.info("Successfully initialized experiment.")
    logger.info("Start initializing runner")

    runner: BaseRunner = runner_cls(exp)
    logger.info("Start running runner")
    runner.run()

def docker_image_exists(image: str) -> bool:
    """Check if a Docker image exists locally.

    Args:
        image (str): image name with tag

    Returns:
        bool
    """
    result = subprocess.run(
        ["docker", "images", "-q", image],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.stdout.strip() != ""

def build_docker_image() -> None:
    """Build docker image using docker-compose in environments/."""
    env_dir = pathlib.Path(__file__).resolve().parent.parent / "environments"
    logger.info(f"[CVRUNNER] Building docker image cvrunner-local from {env_dir}...")
    subprocess.run(["docker-compose", "-f", str(env_dir / "docker-compose.yml"), "build"], check=True)

def run_in_docker(args: Namespace) -> None:
    """Run experiment inside Docker container.

    Args:
        args (Namespace): argurments for training job
    """
    exp_path = args.exp
    extra_args = []
    build = args.build
    image_name = "cvrunner-local:latest"

    # Auto-build if needed
    if build or not docker_image_exists(image_name):
        build_docker_image()

    cmd = [
        "docker", "run", "--rm",
        "-w", "/workspace",
        "-e", "WANDB_API_KEY", # pass through W&B API key
        image_name,  # Docker image name
        "-l",
        "--exp", exp_path
    ] + extra_args

    logger.info(" ".join(cmd))
    subprocess.run(cmd, check=True)

def build_and_push_image(image: str) -> None:
    """Build and push Docker image to registry.

    Args:
        image (str): image name with tag
    """
    project_root = pathlib.Path(__file__).resolve().parent.parent
    logger.info(f"[CVRUNNER] Building and pushing Docker image {image}...")

    subprocess.run([
        "docker", "buildx", "build",
        "--platform", "linux/amd64",
        "-t", image,
        "-f", str(project_root / "environments" / "Dockerfile"),
        str(project_root),
        "--push"
    ], check=True)

def sync_wandb_secret()-> None:
    """Create or update Kubernetes Secret for W&B API key."""
    
    api_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY not set in local environment!")

    # Build YAML manifest inline
    secret_yaml = textwrap.dedent(f"""
    apiVersion: v1
    kind: Secret
    metadata:
      name: wandb-secret
    type: Opaque
    stringData:
      WANDB_API_KEY: "{api_key}"
    """)

    # Pipe YAML directly into kubectl apply
    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=secret_yaml.encode("utf-8"),
        check=True
    )

def run_on_k8s(args: Namespace) -> None:
    """Run experiment on Kubernetes cluster.

    Args:
        args (Namespace): argurments for training job
    """
    image = args.image
    exp_path = args.exp

    if args.build:
        # Always build + push latest image before creating job
        build_and_push_image(image)

    env_dir = pathlib.Path(__file__).resolve().parent.parent / "environments" / "k8s"
    template_file = env_dir / "job-template.yml"

    with open(template_file) as f:
        job = yaml.safe_load(f)

    # Replace placeholders
    job["spec"]["template"]["spec"]["containers"][0]["image"] = image
    job["spec"]["template"]["spec"]["containers"][0]["workingDir"] = "/workspace"
    job["spec"]["template"]["spec"]["containers"][0]["args"] = ["-l", "--exp", exp_path]

    # Inject WANDB_API_KEY from secret
    job["spec"]["template"]["spec"]["containers"][0].setdefault("envFrom", [])
    job["spec"]["template"]["spec"]["containers"][0]["envFrom"].append({
        "secretRef": {"name": "wandb-secret"}
    })

    # Sync or create secret
    sync_wandb_secret()

    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as f:
        yaml.dump(job, f)
        jobfile = f.name

    logger.info(f"[CVRUNNER] Submitting Kubernetes Job using {jobfile}...")
    subprocess.run(["kubectl", "create", "-f", jobfile], check=True)

    logger.info("[CVRUNNER] Tail logs with:")
    logger.info("  kubectl logs -l job-name=$(kubectl get jobs -o jsonpath='{.items[-1:].metadata.name}') -f")

# TODO: support distributed training
def main() -> None:
    """
    Entry point for cvrunner CLI.
    """
    parser = argparse.ArgumentParser(description="Run experiments with cvrunner.")
    parser.add_argument("-e", "--exp", type=str, required=True)
    parser.add_argument("-l", "--run-local", action="store_true", help="Run the experiment locally instead of inside Docker")
    parser.add_argument("--build", action="store_true", help="Build Docker image before running")
    parser.add_argument("--k8s", action="store_true", help="Run on Kubernetes")
    parser.add_argument("--image", type=str, default="oel20/cvrunner:latest", help="Docker image to use")
    args = parser.parse_args()

    if args.run_local:
        run_local(args)
    elif args.k8s:
        run_on_k8s(args)
    else:
        run_in_docker(args)