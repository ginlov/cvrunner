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
from dotenv import dotenv_values

from cvrunner.runner.base_runner import BaseRunner
from cvrunner.experiment.experiment import BaseExperiment
from cvrunner.utils.logger import get_cv_logger

# Add current working directory to sys.path
sys.path.insert(0, str(pathlib.Path.cwd()))
logger = get_cv_logger()

def get_properties(obj):
    props = {}
    for name in dir(obj):
        attr = getattr(type(obj), name, None)
        if isinstance(attr, property):
            props[name] = getattr(obj, name)
    return props

def get_compose_cmd() -> List[str]:
    """Get the docker compose command, either `docker-compose` or `docker compose`.
    Raises:
        RuntimeError: if neither command is found
    Returns:
        List[str]: command as list of strings
    """
    if subprocess.call(["which", "docker-compose"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        return ["docker-compose"]
    try:
        subprocess.run(["docker", "compose", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ["docker", "compose"]
    except subprocess.CalledProcessError:
        raise RuntimeError("Docker Compose not found")

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
        logger.init_wandb(project=wandb_project, run_name=wandb_runname, config=get_properties(exp))

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

def build_docker_image(env_dir) -> None:
    """Build docker image using docker-compose in environments/."""
    logger.info(f"[CVRUNNER] Building docker image cvrunner-local from {env_dir}...")
    subprocess.run(get_compose_cmd() + ["-f", str(env_dir / "docker-compose.yml"), "build"], check=True)

def run_in_docker(args: Namespace) -> None:
    """Run experiment inside Docker container.

    Args:
        args (Namespace): argurments for training job
    """
    exp_path = args.exp
    extra_args = []
    build = args.build
    image_name = "cvrunner-local:latest"

    cwd_env_dir = pathlib.Path.cwd() / "environments"
    if (cwd_env_dir / "docker-compose.yml").exists():
        env_dir = cwd_env_dir
    else:
        # fallback to cvrunner's own environments
        env_dir = pathlib.Path(__file__).resolve().parent.parent / "environments"

    # Auto-build if needed
    if build or not docker_image_exists(image_name):
        build_docker_image(env_dir)

    cmd = get_compose_cmd() + [ 
        "-f", str(env_dir / "docker-compose.yml"),
        "run", "--rm",
        "cvrunner",  # service name from docker-compose.yml
        "-l",
        "--exp", exp_path
    ] + extra_args

    logger.info(" ".join(cmd))
    subprocess.run(cmd, check=True)

def build_and_push_image(image: str, context: str) -> None:
    """Build and push Docker image to registry.

    Args:
        image (str): image name with tag
    """
    env_dir = context / "environments"
    build_docker_image(env_dir)

    logger.info(f"[CVRUNNER] Tagging image cvrunner-local:latest as {image}...")
    subprocess.run(["docker", "tag", "cvrunner-local:latest", image], check=True)
    logger.info(f"[CVRUNNER] Pushing image {image} to registry...")
    subprocess.run(["docker", "push", image], check=True)

def sync_env_secret(context: pathlib.Path) -> None:
    """Create or update Kubernetes Secret from all vars in .env file."""
    env_file = context / ".env"

    if not env_file.exists():
        raise RuntimeError(f".env file not found at {env_file}")

    env_vars = dotenv_values(env_file)
    if not env_vars:
        raise RuntimeError("No variables found in .env file")

    # Build Secret YAML
    secret_yaml = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": "cvrunner-secrets"},
        "type": "Opaque",
        "stringData": {k: str(v) for k, v in env_vars.items() if v is not None},
    }

    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=yaml.dump(secret_yaml).encode("utf-8"),
        check=True,
    )

def run_on_k8s(args: Namespace) -> None:
    """Run experiment on Kubernetes cluster.

    Args:
        args (Namespace): argurments for training job
    """
    image = args.image
    exp_path = args.exp

    # Get project root
    cwd_env_dir = pathlib.Path.cwd()
    if (cwd_env_dir / "environments" / "Dockerfile").exists():
        context = pathlib.Path.cwd()
    else:
        project_root = pathlib.Path(__file__).resolve().parent.parent
        context = project_root

    if args.build:
        # Always build + push latest image before creating job
        logger.info(f"[CVRUNNER] Building and pushing Docker image {image} with context {context}...")
        build_and_push_image(image, context)

    env_dir = context / "environments" / "k8s"
    template_file = env_dir / "job-template.yml"
    
    logger.info(f"[CVRUNNER] Using Kubernetes job template: {template_file}")
    with open(template_file) as f:
        job = yaml.safe_load(f)

    # Replace placeholders
    job["spec"]["template"]["spec"]["containers"][0]["image"] = image
    job["spec"]["template"]["spec"]["containers"][0]["workingDir"] = "/workspace"
    job["spec"]["template"]["spec"]["containers"][0]["args"] = ["-l", "--exp", exp_path]

    # Inject WANDB_API_KEY from secret
    job["spec"]["template"]["spec"]["containers"][0].setdefault("envFrom", [])
    job["spec"]["template"]["spec"]["containers"][0]["envFrom"].append({
        "secretRef": {"name": "cvrunner-secrets"}
    })

    # Sync or create secret
    sync_env_secret(context)

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
