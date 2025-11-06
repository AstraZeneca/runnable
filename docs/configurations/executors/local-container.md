
# Local Container Pipeline Execution

Execute pipelines sequentially using Docker containers for environment isolation - perfect for testing container-based deployments locally.

!!! tip "Container Setup Made Simple"

    Just build a Docker image from your project root - it automatically includes your code, dependencies, and environment!

    ```bash
    docker build -t my-project:latest .
    ```

## Getting Started

### Basic Configuration

```yaml
pipeline-executor:
  type: local-container
  config:
    docker_image: "my-project:latest"
```

## Simple Example

=== "pipeline.py"

    ```python
    from runnable import Pipeline, PythonTask

    def hello_from_container():
        import platform
        print(f"Hello from container running: {platform.platform()}")
        return "success"

    def main():
        task = PythonTask(
            function=hello_from_container,
            name="hello"
        )

        pipeline = Pipeline(steps=[task])
        pipeline.execute()

    if __name__ == "__main__":
        main()
    ```

=== "config.yaml"

    ```yaml
    pipeline-executor:
      type: local-container
      config:
        docker_image: "my-project:latest"
    ```

=== "Run It"

    ```bash
    # Set configuration and run
    RUNNABLE_CONFIGURATION_FILE=config.yaml uv run pipeline.py
    ```

!!! info "Container Isolation"

    Each task runs in a fresh container, giving you clean isolation between steps.

## Why Use Containers Locally?

!!! success "Perfect for Production Testing"

    **Environment reproduction**: Test exactly what runs in production

    - ✅ **Dependency isolation**: Each step gets a clean container environment
    - ✅ **Local validation**: Catch container issues before cloud deployment
    - ✅ **Multiple environments**: Different containers for different pipeline steps

!!! note "Sequential Execution Model"

    - 🔄 **One step at a time**: Runs sequentially like the local executor
    - 🐳 **Container per step**: Each task gets a fresh, isolated container
    - 💻 **Local resources**: Still uses your machine's CPU/memory limits

## Advanced Usage

### Dynamic Container Images

!!! example "Runtime Image Selection"

    Use different images at runtime with environment variables:

    ```yaml
    pipeline-executor:
      type: local-container
      config:
        docker_image: $my_docker_image
    ```

    ```bash
    # Set the image dynamically
    export RUNNABLE_VAR_my_docker_image="my-project:v2.0"
    RUNNABLE_CONFIGURATION_FILE=config.yaml uv run pipeline.py
    ```

### Step-Specific Containers

Different steps can use different container images:

=== "pipeline.py"

    ```python
    from runnable import Pipeline, ShellTask

    def main():
        # Uses default container
        step1 = ShellTask(
            name="python_analysis",
            command="python --version && python analyze.py"
        )

        # Uses specialized R container
        step2 = ShellTask(
            name="r_modeling",
            command="Rscript model.R",
            overrides={"local-container": "r_override"}
        )

        pipeline = Pipeline(steps=[step1, step2])
        pipeline.execute()

    if __name__ == "__main__":
        main()
    ```

=== "config.yaml"

    ```yaml
    pipeline-executor:
      type: local-container
      config:
        docker_image: "my-python:latest"  # Default for most steps
      overrides:
        r_override:
          docker_image: "my-r-env:latest"  # Specialized R environment
    ```

### Debugging Failed Containers

!!! tip "Debug Failed Containers"

    Keep containers around for debugging:

    ```yaml
    pipeline-executor:
      type: local-container
      config:
        docker_image: "my-project:latest"
        auto_remove_container: false  # Keep failed containers
    ```

    Then inspect the failed container:

    ```bash
    # List containers to find the failed one
    docker ps -a

    # Get into the failed container
    docker exec -it <container-id> /bin/bash

    # Or check its logs
    docker logs <container-id>
    ```

## Configuration Reference

::: extensions.pipeline_executor.local_container.LocalContainerExecutor
    options:
        show_root_heading: false
        show_bases: false
        members: false
        show_docstring_description: true
        heading_level: 3

## When to Use Local Container

!!! question "Choose Local Container When"

    - Testing container-based deployments before going to cloud
    - Need environment isolation between pipeline steps
    - Want to replicate production container behavior locally
    - Different steps require different software environments

!!! abstract "Use Regular Local Executor When"

    - Simple development and experimentation
    - All steps use the same environment
    - Want fastest possible execution (no container overhead)

!!! success "Upgrade to Cloud Executors When"

    - Need true parallel execution ([Argo](argo.md))
    - Want distributed compute resources
    - Running production workloads
