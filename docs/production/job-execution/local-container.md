# Local Container Job Execution

Execute jobs in Docker containers on your local machine for environment isolation and consistency - perfect for testing containerized deployments locally.

!!! info "Installation Required"

    Container execution requires the optional Docker dependency:
    ```bash
    pip install runnable[docker]
    ```

!!! tip "Container Setup Made Simple"

    Just build a Docker image from your project root - it automatically includes your code, dependencies, and environment!

    ```bash
    docker build -t my-project:latest .
    ```

## Why Use Container Execution?

!!! success "Container Benefits"

    **Environment isolation**: Clean, reproducible execution environment

    - 🐳 **Consistent environments**: Same container across different machines
    - 📦 **Dependency isolation**: Package everything needed in the image
    - 🔄 **Production parity**: Test with the same image you'll deploy
    - ✅ **Reproducible builds**: Eliminate "works on my machine" issues

!!! note "Trade-offs"

    - 🐳 **Container overhead**: ~1-2 seconds startup time per job
    - 📊 **Docker dependency**: Requires Docker installation and running daemon
    - 💾 **Image management**: Need to build and maintain Docker images

## Getting Started

### Basic Configuration

```yaml
job-executor:
  type: local-container
  config:
    docker_image: "my-project:latest"
```

### Simple Example

=== "job.py"

    ```python
    from runnable import PythonJob
    from examples.common.functions import hello

    def main():
        job = PythonJob(function=hello)
        job.execute()  # Configuration via RUNNABLE_CONFIGURATION_FILE
        return job

    if __name__ == "__main__":
        main()
    ```

=== "config.yaml"

    ```yaml
    job-executor:
      type: local-container
      config:
        docker_image: "my-project:latest"
    ```

**Run the containerized job:**
```bash
# Build your image first
docker build -t my-project:latest .

# Run with container executor
RUNNABLE_CONFIGURATION_FILE=config.yaml uv run job.py
```

!!! info "Container Isolation"

    Each job runs in a fresh container, giving you clean isolation and consistent environments.

## Configuration Reference

```yaml
job-executor:
  type: local-container
  config:
    docker_image: "my-project:latest"  # Required: Docker image to use
    auto_remove_container: true        # Optional: Remove containers after execution
    mock: false                        # Optional: Skip actual execution for testing
    environment:                       # Optional: Environment variables
      DATABASE_URL: "postgresql://localhost/mydb"
      API_KEY: "your-api-key"
```

### Configuration Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `docker_image` | str | **REQUIRED** | Docker image to use for execution |
| `auto_remove_container` | bool | `true` | Automatically remove container after execution |
| `mock` | bool | `false` | Skip execution, simulate success for testing |
| `environment` | dict | `{}` | Environment variables to pass to container |

### Environment Variables

Pass configuration to your containerized jobs:

```yaml
job-executor:
  type: local-container
  config:
    docker_image: "data-pipeline:latest"
    environment:
      DATABASE_URL: "postgresql://localhost/analytics"
      LOG_LEVEL: "INFO"
      BATCH_SIZE: "1000"
```

### Debugging Configuration

Keep containers for inspection:

```yaml
job-executor:
  type: local-container
  config:
    docker_image: "my-project:debug"
    auto_remove_container: false  # Keep container after execution
    mock: false                   # Set to true for workflow testing
```


## Complete Example

=== "python_tasks.py"

    ```python title="examples/11-jobs/python_tasks.py"
    from examples.common.functions import hello
    from runnable import PythonJob

    def main():
        job = PythonJob(function=hello)
        job.execute()
        return job

    if __name__ == "__main__":
        main()
    ```

=== "local-container.yaml"

    ```yaml title="examples/11-jobs/local-container.yaml"
    job-executor:
      type: "local-container"
      config:
        docker_image: demo-runnable-m1:latest
    ```

**Run the example:**
```bash
# Build your image first
docker build -t demo-runnable-m1:latest .

# Run with container configuration
RUNNABLE_CONFIGURATION_FILE=examples/11-jobs/local-container.yaml uv run examples/11-jobs/python_tasks.py
```

## Docker Requirements

Ensure Docker is running and build your image:

```bash
# Check Docker is available
docker --version

# Build your project image
docker build -t my-project:latest .
```

### Simple Dockerfile Example

!!! example "Basic Dockerfile"

    ```dockerfile
    FROM python:3.11

    WORKDIR /app

    # Copy and install dependencies
    COPY requirements.txt .
    RUN pip install -r requirements.txt

    # Copy project code
    COPY . .
    ```

    **Build and use:**
    ```bash
    docker build -t my-project:latest .
    ```

## Troubleshooting

### Common Issues

!!! warning "Module Import Errors"

    **Problem**: `ModuleNotFoundError: No module named 'my_project'`

    **Solution**: Ensure your container environment matches local setup:

    - Check Python version: `python --version`
    - Verify dependencies are installed in container
    - Test imports: `docker run --rm my-project:latest python -c "import my_module"`

!!! warning "Container Exits Immediately"

    **Problem**: Container stops without running the job

    **Solution**:

    - Verify Docker image has required dependencies
    - Test basic functionality: `docker run --rm my-project:latest python -c "from runnable import PythonJob"`

### Debug Mode

Keep containers for inspection:

```yaml
job-executor:
  type: local-container
  config:
    docker_image: "my-project:latest"
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

## When to Upgrade

Consider other executors when you need:

!!! abstract "No Docker Dependency"

    **[Local](local.md)**: For simple development without container overhead

!!! success "Production Orchestration"

    **[Kubernetes](kubernetes.md)**: For production workloads with resource management and scaling

---

**Related:** [Pipeline Container Execution](../pipeline-execution/local-container.md) | [All Job Executors](overview.md)
