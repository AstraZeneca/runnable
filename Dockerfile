# Test Dockerfile with all runtime dependencies
# Apple M1 compatible multi-platform image

FROM python:3.11-slim

# Set working directory
WORKDIR /app

USER root

# Install system dependencies including CA certificates
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Install uv for fast dependency management
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock README.md ./

# Configure UV for better SSL handling
ENV UV_NATIVE_TLS=1

RUN uv sync --all-extras --frozen --all-groups --no-group torchexamples

COPY runnable/ ./runnable/
COPY extensions/ ./extensions/
COPY examples/ ./examples/

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"
