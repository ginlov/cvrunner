# Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (example, adapt as needed)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for caching
COPY pyproject.toml poetry.lock* requirements.txt* ./

# Install dependencies
RUN pip install --no-cache-dir -U pip \
    && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi \
    && if [ -f poetry.lock ]; then pip install poetry && poetry install --no-dev; fi

# Copy project files
COPY . ./cvrunner

# Install cvrunner
RUN pip install -e ./cvrunner

# Set default entrypoint to cvrunner CLI
ENTRYPOINT ["cvrunner"]
