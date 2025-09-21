# Choose base image with CUDA runtime but minimal extras
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install minimal system dependencies + Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-distutils python3-pip \
    libjpeg-dev libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# Ensure python3 points to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --no-cache-dir -U pip

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121\
    --extra-index-url https://download.pytorch.org/whl/cu121

# Set working directory
WORKDIR /app

# Copy project files
COPY . ./cvrunner

# Install cvrunner in editable mode
RUN pip install -e ./cvrunner

# Set default entrypoint to cvrunner CLI
ENTRYPOINT ["cvrunner"]
