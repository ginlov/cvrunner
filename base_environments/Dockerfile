# Dockerfile
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set working directory
WORKDIR /app

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
