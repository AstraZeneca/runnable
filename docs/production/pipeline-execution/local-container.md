
# Local Container Pipeline Execution

Execute pipelines using Docker containers with optional parallel processing - perfect for testing container-based deployments locally with environment isolation.

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

**Run the pipeline:**
```bash
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

!!! note "Execution Models"

    **Sequential (Default)**:

    - 🔄 **One step at a time**: Tasks run sequentially for simplicity
    - 🐳 **Container per step**: Each task gets a fresh, isolated container
    - 💻 **Local resources**: Uses your machine's CPU/memory limits

    **Parallel (Optional)**:

    - ⚡ **Parallel branches**: `parallel` and `map` nodes can run simultaneously
    - 🐳 **Multiple containers**: Each branch gets its own container
    - 📋 **Requires compatible run log store**: Use `chunked-fs` for parallel writes

## Parallel Execution

Enable parallel processing for container-based workflows:

=== "pipeline.py"

    ```python
    from runnable import Pipeline, PythonTask, Parallel

    def process_in_container(data_chunk):
        import platform
        print(f"Processing chunk {data_chunk} on {platform.platform()}")
        return f"processed_{data_chunk}"

    def main():
        # Parallel branches that run in separate containers
        parallel_node = Parallel(
            name="container_parallel",
            branches={
                "process_a": [PythonTask(function=process_in_container, name="task_a")],
                "process_b": [PythonTask(function=process_in_container, name="task_b")],
                "process_c": [PythonTask(function=process_in_container, name="task_c")]
            }
        )

        pipeline = Pipeline(steps=[parallel_node])

        # Execute with parallel container support
        pipeline.execute(configuration_file="parallel_container.yaml")

        return pipeline

    if __name__ == "__main__":
        main()
    ```

=== "parallel_container.yaml"

    ```yaml
    pipeline-executor:
      type: local-container
      config:
        docker_image: "my-project:latest"
        enable_parallel: true

    # Required for parallel execution
    run-log-store:
      type: chunked-fs

    catalog:
      type: file-system
    ```

**Run with parallel containers:**
```bash
# Build your image first
docker build -t my-project:latest .

# Execute the pipeline
uv run pipeline.py
```


!!! success "Parallel Container Benefits"

    - **True isolation**: Each parallel branch runs in its own container
    - **Resource utilization**: Uses multiple CPU cores simultaneously
    - **Production testing**: Test parallel behavior before deploying to Kubernetes

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

Different steps can use different container images - useful when you need specialized environments for different parts of your pipeline.

**How it works:**

1. **Define multiple configurations** in your config file using `overrides`
2. **Reference the override** in your task using the `overrides` parameter
3. **Each task runs** in its specified container environment

=== "pipeline.py"

    ```python
    from runnable import Pipeline, ShellTask

    def main():
        # Uses default Python container (from main config)
        step1 = ShellTask(
            name="python_analysis",
            command="python --version && python analyze.py"
        )

        # Uses specialized R container (from "r_override" configuration)
        step2 = ShellTask(
            name="r_modeling",
            command="Rscript model.R",
            overrides={"local-container": "r_override"}  # References config below
        )

        pipeline = Pipeline(steps=[step1, step2])
        pipeline.execute()

    if __name__ == "__main__":
        main()
    ```

    !!! info "Understanding the Override"

        `overrides={"local-container": "r_override"}` means:

        - **"local-container"**: The executor type we're overriding
        - **"r_override"**: The name of the override configuration (defined in config.yaml)
        - **Result**: This task will use the R container instead of the default Python container

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

```yaml
pipeline-executor:
  type: local-container
  config:
    docker_image: "my-project:latest"  # Required: Docker image to use
    enable_parallel: false             # Enable parallel execution
    auto_remove_container: true        # Remove containers after execution
    environment:                       # Environment variables for containers
      VAR_NAME: "value"
    overrides:                        # Step-specific configurations
      alt_config:
        docker_image: "alternative:latest"
        auto_remove_container: false
        environment:
          SPECIAL_VAR: "special_value"
```

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
