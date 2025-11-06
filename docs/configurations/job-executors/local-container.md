# Local Container Job Execution

Execute jobs in Docker containers on your local machine for environment isolation and consistency.

!!! tip "Environment Parity Made Simple"

    **Easy Setup**: Your container just needs to match your local development environment. This is straightforward - simply build from your project root!

    ```bash
    # From your project root directory
    docker build -t my-project:latest .
    ```

    **Your container automatically gets**:

    - Same Python version and dependencies (via requirements.txt)
    - Identical project structure (via COPY . .)
    - All required packages and modules
    - Working directory that matches your local setup

    **Result**: Container works exactly like your local environment! 🎉

??? example "Simple Dockerfile for environment parity"

    ```dockerfile title="Dockerfile"
    # Use the same Python version you're running locally
    FROM python:3.11

    # Set working directory
    WORKDIR /app

    # Copy and install dependencies
    COPY requirements.txt .
    RUN pip install -r requirements.txt

    # Copy your entire project (maintains structure)
    COPY . .

    # That's it! Container now matches your local environment
    ```

    ```bash
    # Build and use
    docker build -t my-project:latest .

    # Configure Runnable to use it
    export RUNNABLE_CONFIGURATION_FILE=container.yaml
    python my_job.py
    ```

## When to Use

- **Environment isolation**: Separate from your local Python environment
- **Consistent execution**: Same container across different machines
- **Dependency management**: Package everything needed in the image
- **Production parity**: Test with the same image you'll deploy
- **Reproducible builds**: Eliminate "works on my machine" issues

## Quick Start

```python
from runnable import PythonJob
from examples.common.functions import hello

job = PythonJob(function=hello)
job.execute()  # Configuration via RUNNABLE_CONFIGURATION_FILE
```

```bash
# Recommended: Set configuration via environment variable
export RUNNABLE_CONFIGURATION_FILE=local-container.yaml
python my_job.py

# Or inline for specific runs
RUNNABLE_CONFIGURATION_FILE=local-container.yaml python my_job.py
```

## Essential Configuration

### Required Fields

The **only required field** is the Docker image:

```yaml title="local-container.yaml"
job-executor:
  type: local-container
  config:
    docker_image: "my-data-pipeline:latest"  # REQUIRED - your project-specific image
```

This minimal configuration:

- Runs your job in your custom project container
- Auto-removes container after execution
- Mounts necessary volumes for logs and data

### Project-Specific Image

```yaml
job-executor:
  type: local-container
  config:
    docker_image: "fraud-detection-model:v2.1.0"
```

## Common Customizations

### Environment Variables

```yaml
job-executor:
  type: local-container
  config:
    docker_image: "customer-segmentation:latest"
    environment:
      DATABASE_URL: "postgresql://localhost/customer_db"
      ML_MODEL_VERSION: "v3.2.1"
      FEATURE_STORE_ENDPOINT: "https://features.company.com"
```

### Container Management

```yaml
job-executor:
  type: local-container
  config:
    docker_image: "recommendation-engine:debug"
    auto_remove_container: false  # Keep container for debugging
    mock: false  # Set to true for testing
```

## All Configuration Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `docker_image` | str | **REQUIRED** | Docker image to use for execution |
| `auto_remove_container` | bool | `true` | Automatically remove container after execution |
| `mock` | bool | `false` | Skip execution, simulate success |
| `environment` | dict | `{}` | Environment variables to pass to container |


## Complete Example

=== "Python Code"

    ```python title="examples/11-jobs/python_tasks.py"
    --8<-- "examples/11-jobs/python_tasks.py"
    ```

=== "Container Config"

    ```yaml title="examples/11-jobs/local-container.yaml"
    --8<-- "examples/11-jobs/local-container.yaml"
    ```

=== "Run It"

    ```bash
    # Recommended: Environment variable approach
    export RUNNABLE_CONFIGURATION_FILE=examples/11-jobs/local-container.yaml
    uv run examples/11-jobs/python_tasks.py

    # Alternative: Inline config flag
    uv run examples/11-jobs/python_tasks.py --config examples/11-jobs/local-container.yaml
    ```

## Docker Requirements

Ensure Docker is running:

```bash
# Check Docker is available
docker --version

# Build or pull your project-specific image
docker build -t my-analytics-project:latest .
# OR pull from your registry
docker pull your-registry.com/analytics-team/data-pipeline:v1.0.0
```

## Building Environment-Matched Containers

**Critical**: Your container must replicate your local development environment exactly.

### Requirements Checklist

Before building your container, ensure:

- [ ] **Python Version**: Exact match to your local Python version
- [ ] **Dependencies**: All packages from your local environment
- [ ] **Project Structure**: Same directory layout and file paths
- [ ] **Working Directory**: Container WORKDIR matches your project root
- [ ] **Import Paths**: All modules importable in the same way

### Building Matched Containers

```dockerfile title="Dockerfile"
# Match your exact Python version from local development
FROM python:3.11.9-slim

# Install system dependencies your project needs
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to match your local project structure
WORKDIR /app/ml-fraud-detection

# Copy and install exact dependencies (should match your local environment)
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project source code maintaining structure
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Ensure Python can find your modules (match local PYTHONPATH)
ENV PYTHONPATH=/app/ml-fraud-detection/src:$PYTHONPATH

# Set project-specific environment
ENV PROJECT_NAME=fraud_detection
ENV MODEL_VERSION=v2.1.0
```

### Verify Environment Parity

Test your container matches your local setup:

```bash
# Build your project-specific image
docker build -t fraud-detection-pipeline:latest .

# Test import paths work the same way as locally
docker run --rm fraud-detection-pipeline:latest python -c "
import sys
print('Python version:', sys.version)
print('Python path:', sys.path)

# Test your project imports (replace with your actual modules)
from src.fraud_detection.models import risk_scorer
from src.fraud_detection.features import feature_extractor
print('Project imports successful')
"

# Test runnable works with your project
docker run --rm fraud-detection-pipeline:latest python -c "
from runnable import PythonJob
from src.fraud_detection.jobs import score_transactions
print('Runnable + project integration successful')
"
```

### Configuration

```yaml
job-executor:
  type: local-container
  config:
    docker_image: "fraud-detection-pipeline:latest"
```

## Advanced Configuration

??? example "Production-like setup"

    ```yaml
    job-executor:
      type: local-container
      config:
        docker_image: "customer-churn-predictor:v2.3.1"
        auto_remove_container: false  # Keep for debugging
        environment:
          # Database connections
          POSTGRES_URL: "postgresql://analytics-db:5432/customer_data"
          REDIS_URL: "redis://cache:6379/0"
          # ML model configuration
          MODEL_REGISTRY_URL: "https://mlflow.company.com"
          FEATURE_STORE_URL: "https://feast.company.com"
          # Project settings
          LOG_LEVEL: "INFO"
          EXPERIMENT_TRACKING: "enabled"
    ```

## Troubleshooting

### Common Issues

**Environment Mismatch Errors**:
```
ModuleNotFoundError: No module named 'my_project'
ImportError: cannot import name 'my_function'
```
**Solution**: Your container environment doesn't match local setup
- Verify Python version matches exactly (`python --version`)
- Check all dependencies are installed in container
- Ensure WORKDIR and PYTHONPATH are correctly set
- Test imports work: `docker run --rm fraud-detection-pipeline:latest python -c "from src.models import predictor"`

**Container exits immediately**:
- Check your Docker image has the required Python dependencies
- Verify your function imports work in the container
- Test with: `docker run --rm customer-analytics:v1.0 python -c "from runnable import PythonJob"`

**Function not found errors**:
```
AttributeError: module 'my_module' has no attribute 'my_function'
```
**Solution**: Project structure mismatch
- Ensure your code is copied to the same path structure in container
- Check WORKDIR matches your local project root
- Verify all source files are included in the container

**Permission errors**:
- Ensure Docker daemon is running
- Check user has Docker permissions (`docker run hello-world`)

**Volume mount issues**:
- Volumes are handled automatically - no manual configuration needed
- Check Docker has access to your working directory

### Debug Mode

Keep container for inspection:

```yaml
job-executor:
  type: local-container
  config:
    docker_image: "python:3.11"
    auto_remove_container: false  # Container stays after execution
```

```bash
# List containers after job runs
docker ps -a

# Inspect logs
docker logs <container-id>

# Interactive shell
docker exec -it <container-id> /bin/bash
```

## Performance Considerations

- **Startup overhead**: ~1-2 seconds container creation
- **Image size**: Smaller images start faster
- **Volume mounting**: Local filesystem access through Docker
- **Resource usage**: Container memory/CPU limits via Docker

## When to Graduate

Consider other executors when you need:

- **Production orchestration**: → [Kubernetes](kubernetes.md)
- **No Docker dependency**: → [Local](local.md)
- **Resource management**: → [Kubernetes](kubernetes.md)
- **Multi-node execution**: → [Kubernetes](kubernetes.md)

---

**Related:** [Pipeline Container Execution](../executors/local-container.md) | [All Job Executors](overview.md)
