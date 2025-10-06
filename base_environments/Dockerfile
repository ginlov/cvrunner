# ---------- Stage 1: Builder ----------
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

# Install Python 3.10 and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-distutils python3-pip \
    libjpeg-dev libpng-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# Ensure python3 points to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip and build helpers
RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel

# Install PyTorch + deps (builder has them, will be copied later)
RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Set deterministic env vars
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV PYTHONHASHSEED=0

# Set working directory
WORKDIR /app

# Copy project files
COPY . ./cvrunner

# Install your project
RUN pip install -e ./cvrunner


# ---------- Stage 2: Runtime ----------
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

# Install Python 3.10 (minimal, no build deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-distutils python3-pip \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 points to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app (already installed in builder, but this ensures CLI works)
WORKDIR /app
COPY --from=builder /app /app

# Default entrypoint
ENTRYPOINT ["cvrunner"]
