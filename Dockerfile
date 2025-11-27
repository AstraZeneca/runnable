# Test Dockerfile with all runtime dependencies
# Apple M1 compatible multi-platform image

FROM python:3.11-slim

# Set working directory
WORKDIR /app

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --all-extras --frozen --all-groups

COPY runnable/ ./runnable/
COPY extensions/ ./extensions/
COPY examples/ ./examples/

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"
