# 🌍 Deploy Anywhere

Here's the ultimate superpower: Same code runs everywhere. Just change the configuration.

**Good news**: Your runnable pipeline is already production-ready. You've been building with production in mind this whole time.

## Your code never changes

```python
from runnable import Pipeline, PythonTask

def train_model():
    # Your model training logic
    print("Training model...")
    return "model_v1.pkl"

def main():
    pipeline = Pipeline(steps=[
        PythonTask(function=train_model, name="training")
    ])

    pipeline.execute()
    return pipeline

if __name__ == "__main__":
    main()
```

## Change where it runs

### 💻 Local development
```python
pipeline.execute()  # Runs on your laptop
```

### 🐳 Container execution
```python
pipeline.execute(configuration_file="local-container.yaml")  # Runs in Docker
```

### ☁️ Cloud platforms
```python
pipeline.execute(configuration_file="argo.yaml")  # Runs on Argo Workflows
```

## Configuration files define the environment

The same pipeline code runs in any environment - just change the configuration:

=== "Development"

    **Fast local development** (default configuration):
    ```yaml title="local.yaml"
    pipeline-executor:
      type: local

    run-log-store:
      type: file-system

    catalog:
      type: file-system

    secrets:
      type: env-secrets
    ```

    ```bash
    # Uses default configuration
    uv run my_pipeline.py
    ```

=== "Testing"

    **Pipeline validation** without execution:
    ```yaml title="mocked.yaml"
    pipeline-executor:
      type: mocked

    run-log-store:
      type: file-system

    catalog:
      type: file-system

    secrets:
      type: env-secrets
    ```

    ```bash
    export RUNNABLE_CONFIGURATION_FILE=mocked.yaml
    uv run my_pipeline.py
    ```

=== "Containerized"

    **Isolated execution** in containers:
    ```yaml title="local-container.yaml"
    pipeline-executor:
      type: local-container
      config:
        # Build from project root to include runnable + dependencies
        docker_image: "my-pipeline:latest"  # or use existing image with runnable installed

    run-log-store:
      type: file-system

    catalog:
      type: file-system

    secrets:
      type: env-secrets
    ```

    !!! info "Container Image Requirements"

        **The Docker image must have Runnable installed**. Either:

        - **Build from your project root**: `docker build -t my-pipeline:latest .` (includes your code + runnable)
        - **Use a base image with runnable**: `FROM python:3.11` then `RUN pip install runnable`
        - **Never use bare `python:3.11`** - it doesn't include runnable

    ```bash
    export RUNNABLE_CONFIGURATION_FILE=local-container.yaml
    uv run my_pipeline.py
    ```

=== "Production"

    **Production orchestration** on Kubernetes:
    ```yaml title="argo.yaml"
    pipeline-executor:
      type: argo
      config:
        pvc_for_runnable: runnable
        defaults:
          image: "my-pipeline:v1.0"
          resources:
            limits:
              cpu: "2"
              memory: 4Gi
            requests:
              cpu: "1"
              memory: 2Gi
        argoWorkflow:
          metadata:
            generateName: "pipeline-"
            namespace: production
          spec:
            serviceAccountName: "pipeline-executor"

    run-log-store:
      type: chunked-fs
      config:
        log_folder: /mnt/run_log_store

    catalog:
      type: s3
      config:
        bucket: production-data

    secrets:
      type: env-secrets
    ```

    ```bash
    export RUNNABLE_CONFIGURATION_FILE=argo.yaml
    uv run my_pipeline.py
    ```

**Same code, different environments** - just change the `RUNNABLE_CONFIGURATION_FILE`.

## The power of environment-agnostic code

**Development:**
```bash
# Quick local testing
uv run my_pipeline.py
```

**Production:**
```bash
# Same code, production environment
export RUNNABLE_CONFIGURATION_FILE=production-argo.yaml
uv run my_pipeline.py
```

## Real-world example

From development to production:

```python
from runnable import Pipeline, PythonTask
from examples.common.functions import hello

def main():
    # Same code, different environments
    task = PythonTask(function=hello, name="say_hello")
    pipeline = Pipeline(steps=[task])

    # This execute() call works for both development and production
    # Environment determined by RUNNABLE_CONFIGURATION_FILE
    pipeline.execute()
    return pipeline

if __name__ == "__main__":
    main()
```

**Development** (uses default local config):
```bash
uv run my_pipeline.py
```

**Production** (same code, argo config):
```bash
export RUNNABLE_CONFIGURATION_FILE=argo.yaml
uv run my_pipeline.py
```

??? example "See complete runnable code"
    ```python title="examples/01-tasks/python_tasks.py"
    --8<-- "examples/01-tasks/python_tasks.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/01-tasks/python_tasks.py
    ```

**Dev environment:** Runs in 2 seconds on your laptop
**Prod environment:** Runs on Kubernetes with monitoring, logging, and auto-scaling

**Zero code changes.**


## Configuration Reference

Ready to customize your deployment? Check the configuration documentation:

- **[Pipeline Executors](pipeline-execution/overview.md)** - Choose where pipelines run (local, argo, etc.)
- **[Job Executors](job-execution/overview.md)** - Configure task execution (local, containers, kubernetes)
- **[Storage](catalog.md)** - Set up data persistence (local, S3, MinIO)
- **[Logging](run-log.md)** - Configure execution logs
- **[Secrets](secrets.md)** - Manage sensitive configuration

You now know the core concepts! Start with the examples above, then dive into the configuration reference for advanced setups.
