import argparse
import sys
import pathlib
import subprocess
import os
import tempfile
import yaml
import torch
import warnings
from dotenv import dotenv_values
from argparse import Namespace

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore", module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="wandb.analytics.sentry")

# Add current working directory to sys.path
sys.path.insert(0, str(pathlib.Path.cwd()))

from cvrunner.utils.logger import get_cv_logger

logger = get_cv_logger()

# --- HELPER: Docker Compose ---
def get_compose_cmd():
    """Get the docker compose command."""
    if subprocess.call(["which", "docker-compose"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        return ["docker-compose"]
    try:
        subprocess.run(["docker", "compose", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ["docker", "compose"]
    except subprocess.CalledProcessError:
        raise RuntimeError("Docker Compose not found")

def docker_image_exists(image: str) -> bool:
    """Check if a Docker image exists locally."""
    result = subprocess.run(
        ["docker", "images", "-q", image],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return result.stdout.strip() != ""

def build_docker_image(env_dir) -> None:
    logger.info(f"[CVRUNNER] Building docker image cvrunner-local from {env_dir}...")
    subprocess.run(get_compose_cmd() + ["-f", str(env_dir / "docker-compose.yml"), "build"], check=True)

# --- CORE LOGIC: Local Run ---

def run_local(args: Namespace) -> None:
    """
    Run experiment locally using torchrun.
    This handles both Single-GPU and Multi-GPU transparently.
    """
    # 1. Detect Hardware
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"🚀 Found {num_gpus} GPUs. Preparing distributed run...")
    else:
        num_gpus = 0
        logger.warning("⚠️ No GPUs detected. Running on CPU (slow).")

    # 2. Resolve Experiment Path
    exp_path = pathlib.Path(args.exp).resolve()
    if not exp_path.exists():
        logger.error(f"Experiment file not found: {exp_path}")
        sys.exit(1)

    # 3. Construct torchrun command
    # We use 'torchrun' to launch the DistributedTrainRunner script
    cmd = [
        "torchrun",
        "--nproc_per_node", str(num_gpus if num_gpus > 0 else 1),
        "--rdzv_backend", "c10d",
        "--rdzv_endpoint", "localhost:0", # Let OS pick a random free port
        "-m", "cvrunner.distributed_train_script", # <--- The script we wrote
        "--experiment_path", str(exp_path)
    ]

    logger.info(f"Executing: {' '.join(cmd)}")

    # 4. Execute
    try:
        # We allow stdout/stderr to stream directly to the console
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        logger.info("\n🛑 Training interrupted by user.")
        sys.exit(130)

# --- DOCKER RUN ---

def run_in_docker(args: Namespace) -> None:
    """Run experiment inside Docker container."""
    exp_path = args.exp
    build = args.build
    image_name = "cvrunner-local:latest"

    cwd_env_dir = pathlib.Path.cwd() / "environments"
    if (cwd_env_dir / "docker-compose.yml").exists():
        env_dir = cwd_env_dir
    else:
        # fallback to cvrunner's own environments
        env_dir = pathlib.Path(__file__).resolve().parent.parent / "environments"

    if build or not docker_image_exists(image_name):
        build_docker_image(env_dir)

    # Note: We pass '-l' inside the container so the container runs 'run_local' (torchrun) internally
    cmd = get_compose_cmd() + [ 
        "-f", str(env_dir / "docker-compose.yml"),
        "run", "--rm",
        "cvrunner", 
        "-l", # <--- Important: Trigger local run logic inside container
        "--exp", exp_path
    ]

    logger.info(" ".join(cmd))
    subprocess.run(cmd, check=True)

# --- K8S RUN ---

def build_and_push_image(image: str, context: pathlib.Path) -> None:
    env_dir = context / "environments"
    build_docker_image(env_dir)
    logger.info(f"[CVRUNNER] Tagging and Pushing {image}...")
    subprocess.run(["docker", "tag", "cvrunner-local:latest", image], check=True)
    subprocess.run(["docker", "push", image], check=True)

def sync_env_secret(context: pathlib.Path) -> None:
    env_file = context / ".env"
    if not env_file.exists():
        return # Optional

    env_vars = dotenv_values(env_file)
    secret_yaml = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": "cvrunner-secrets"},
        "type": "Opaque",
        "stringData": {k: str(v) for k, v in env_vars.items() if v is not None},
    }
    subprocess.run(["kubectl", "apply", "-f", "-"], input=yaml.dump(secret_yaml).encode("utf-8"), check=True)

def run_on_k8s(args: Namespace) -> None:
    """Submit Job to K8s."""
    image = args.image
    exp_path = args.exp
    
    # ... (Context resolution logic same as before) ...
    cwd_env_dir = pathlib.Path.cwd()
    if (cwd_env_dir / "environments" / "Dockerfile").exists():
        context = pathlib.Path.cwd()
    else:
        context = pathlib.Path(__file__).resolve().parent.parent

    if args.build:
        build_and_push_image(image, context)

    env_dir = context / "environments" / "k8s"
    template_file = env_dir / "job-template.yml"
    
    with open(template_file) as f:
        job = yaml.safe_load(f)

    # Update Job Spec
    container = job["spec"]["template"]["spec"]["containers"][0]
    container["image"] = image
    container["workingDir"] = "/workspace"
    
    # CRITICAL: K8s args now also just trigger the local run inside the pod
    # Because the pod environment (Distributed) matches local torchrun
    container["args"] = ["-l", "--exp", exp_path]

    # Inject Secrets
    container.setdefault("envFrom", []).append({"secretRef": {"name": "cvrunner-secrets"}})
    sync_env_secret(context)

    # Submit
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as f:
        yaml.dump(job, f)
        jobfile = f.name

    logger.info(f"[CVRUNNER] Submitting Job...")
    subprocess.run(["kubectl", "create", "-f", jobfile], check=True)
    logger.info("[CVRUNNER] Job submitted.")

# --- MAIN ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiments with cvrunner.")
    parser.add_argument("-e", "--exp", type=str, required=True)
    parser.add_argument("-l", "--run-local", action="store_true", help="Run locally")
    parser.add_argument("--build", action="store_true", help="Build Docker image")
    parser.add_argument("--k8s", action="store_true", help="Run on Kubernetes")
    parser.add_argument("--image", type=str, default="oel20/cvrunner:latest")
    args = parser.parse_args()

    if args.run_local:
        run_local(args)
    elif args.k8s:
        run_on_k8s(args)
    else:
        run_in_docker(args)

if __name__ == "__main__":
    main()
