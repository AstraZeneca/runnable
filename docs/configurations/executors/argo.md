# Argo Workflows Pipeline Execution

Scale your pipelines to the cloud with true parallel execution using [Argo Workflows](https://argo-workflows.readthedocs.io/en/latest/) - the most powerful execution environment for production ML workflows.

!!! success "True Parallel Execution"

    **Finally, full parallelization!** Unlike local executors, Argo runs your `parallel` and `map` nodes simultaneously across multiple pods, dramatically speeding up your pipelines.

## Getting Started

!!! tip "Simple Cloud Setup"

    Runnable generates standard Argo workflow YAML - your infrastructure team can deploy it using existing Kubernetes and Argo tooling!

### Basic Configuration

```yaml
pipeline-executor:
  type: argo
  config:
    image: "my-pipeline:latest"  # Your containerized pipeline
    output_file: "workflow.yaml"  # Generated Argo workflow
```


### Simple Example

=== "pipeline.py"

    ```python
    from runnable import Pipeline, PythonTask, Parallel

    def process_data_a():
        print("Processing dataset A...")
        # Your ML logic here
        return {"accuracy": 0.95}

    def process_data_b():
        print("Processing dataset B...")
        # Your ML logic here
        return {"accuracy": 0.92}

    def combine_results(results_a, results_b):
        print(f"A: {results_a['accuracy']}, B: {results_b['accuracy']}")
        return {"best": max(results_a['accuracy'], results_b['accuracy'])}

    def main():
        # These run in parallel in Argo!
        parallel_processing = Parallel(
            name="process_datasets",
            branches={
                "dataset_a": PythonTask(function=process_data_a, name="process_a"),
                "dataset_b": PythonTask(function=process_data_b, name="process_b")
            }
        )

        combine_task = PythonTask(
            function=combine_results,
            name="combine"
        )

        pipeline = Pipeline(steps=[parallel_processing, combine_task])
        pipeline.execute()  # Generates workflow.yaml

    if __name__ == "__main__":
        main()
    ```

=== "config.yaml"

    ```yaml
    pipeline-executor:
      type: argo
      config:
        image: "my-pipeline:latest"
        output_file: "workflow.yaml"
    ```

=== "Run It"

    ```bash
    # Generate Argo workflow
    RUNNABLE_CONFIGURATION_FILE=config.yaml uv run pipeline.py

    # Deploy to your Kubernetes cluster (via your infrastructure team)
    kubectl apply -f workflow.yaml
    ```

!!! info "Parallel Execution"

    In this example, `process_data_a` and `process_data_b` run simultaneously on different Kubernetes pods, then `combine_results` runs after both complete.

## Why Use Argo Workflows?

!!! success "Cloud-Scale Benefits"

    **True parallelization**: `parallel` and `map` nodes run simultaneously

    - ⚡ **Faster pipelines**: Utilize multiple CPU cores across pods
    - 🔄 **Elastic scaling**: Kubernetes automatically manages resources
    - 🏗️ **Production ready**: Battle-tested in enterprise environments
    - 📊 **Rich monitoring**: Native Kubernetes and Argo UI integration

!!! note "Trade-offs"

    - ⚙️ **Infrastructure requirement**: Needs Kubernetes cluster with Argo installed
    - 🐳 **Container overhead**: Each task runs in separate pods
    - 🔧 **Setup complexity**: More moving parts than local executors

## Advanced Features

### Dynamic Parameters

!!! example "Runtime Parameter Control"

    Make your workflows configurable by exposing parameters to the Argo UI:

    ```yaml
    pipeline-executor:
      type: argo
      config:
        image: "my-pipeline:latest"
        output_file: "workflow.yaml"
        expose_parameters_as_inputs: true  # Enable parameter inputs
    ```

    Now parameters become configurable in the Argo UI at runtime!

### Storage and Persistence

!!! example "Shared Storage Between Tasks"

    Use persistent volumes to share data between tasks:

    ```yaml
    pipeline-executor:
      type: argo
      config:
        image: "my-pipeline:latest"
        persistent_volumes:
          - name: shared-data
            mount_path: /shared
    ```

### Kubernetes Secrets

!!! tip "Secure Credential Management"

    Access cluster secrets in your pipeline tasks:

    ```yaml
    pipeline-executor:
      type: argo
      config:
        image: "my-pipeline:latest"
        secrets_from_k8s:
          - environment_variable: DB_CONNECTION
            secret_name: database-credentials
            secret_key: connection_string
    ```


### Resource Management

!!! example "Custom Resource Requirements"

    Different tasks can have different compute requirements:

    ```yaml
    pipeline-executor:
      type: argo
      config:
        image: "my-pipeline:latest"
        # Default resources
        resources:
          requests:
            memory: "1Gi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "500m"
        # Step-specific overrides
        overrides:
          gpu_training:
            resources:
              requests:
                memory: "8Gi"
                cpu: "2"
                nvidia.com/gpu: "1"
              limits:
                memory: "16Gi"
                cpu: "4"
                nvidia.com/gpu: "1"
    ```

    Then use the override in your pipeline:

    ```python
    gpu_task = PythonTask(
        function=train_model,
        name="train_with_gpu",
        overrides={"argo": "gpu_training"}
    )
    ```

### Parallelism Control

!!! tip "Manage Resource Usage"

    Control how many tasks run simultaneously to avoid overwhelming your cluster:

    ```yaml
    pipeline-executor:
      type: argo
      config:
        image: "my-pipeline:latest"
        parallelism: 5  # Max 5 tasks running at once
        overrides:
          sequential_processing:
            parallelism: 1  # Force sequential execution
    ```

### Node Selection

!!! example "Target Specific Nodes"

    Run tasks on specific node types (e.g., GPU nodes):

    ```yaml
    pipeline-executor:
      type: argo
      config:
        image: "my-pipeline:latest"
        overrides:
          gpu_nodes:
            node_selector:
              accelerator: "nvidia-tesla-k80"
            tolerations:
              - key: "gpu"
                operator: "Equal"
                value: "true"
                effect: "NoSchedule"
    ```


## Configuration Reference

::: extensions.pipeline_executor.argo.ArgoExecutor
    options:
        show_root_heading: false
        show_bases: false
        members: false
        show_docstring_description: true
        heading_level: 3

## Production Considerations

!!! warning "Infrastructure Requirements"

    **Before using Argo**: Ensure your Kubernetes cluster has Argo Workflows installed and configured

!!! tip "Service Compatibility"

    **Storage**: Use shared storage (persistent volumes) for `catalog` and `run_log_store` - the `buffered` run log store won't work across pods

    **Secrets**: Use Kubernetes secrets via `secrets_from_k8s` rather than `.env` files


## Complete Production Example

!!! example "Full Configuration with Best Practices"

    ```yaml
    # production-argo-config.yaml
    pipeline-executor:
      type: argo
      config:
        # Core settings
        image: "my-pipeline:v1.2.3"
        output_file: "workflow.yaml"

        # Runtime parameters
        expose_parameters_as_inputs: true

        # Storage
        persistent_volumes:
          - name: shared-storage
            mount_path: /shared

        # Security
        secrets_from_k8s:
          - environment_variable: DB_CONNECTION
            secret_name: database-credentials
            secret_key: connection_string

        # Resource management
        parallelism: 10
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1"

        # Task-specific overrides
        overrides:
          gpu_training:
            resources:
              requests:
                nvidia.com/gpu: "1"
                memory: "8Gi"
                cpu: "2"
              limits:
                nvidia.com/gpu: "1"
                memory: "16Gi"
                cpu: "4"
            node_selector:
              accelerator: "nvidia-tesla-v100"

          lightweight_tasks:
            resources:
              requests:
                memory: "512Mi"
                cpu: "100m"
              limits:
                memory: "1Gi"
                cpu: "200m"

    # Supporting services
    run_log_store:
      type: file-system
      config:
        log_folder: /shared/logs

    catalog_handler:
      type: file-system
      config:
        catalog_location: /shared/data
    ```

### Deployment Workflow

1. **Generate workflow**: `RUNNABLE_CONFIGURATION_FILE=production-argo-config.yaml uv run my_pipeline.py`
2. **Review generated YAML**: Check `workflow.yaml` for correctness
3. **Deploy to cluster**: `kubectl apply -f workflow.yaml`
4. **Monitor execution**: Use Argo UI or `kubectl` to track progress

!!! tip "CI/CD Integration"

    In production, integrate this into your CI/CD pipeline to automatically generate and deploy workflows when your pipeline code changes.


## When to Use Argo Workflows

!!! question "Choose Argo When"

    - Need true parallel execution of `parallel` and `map` nodes
    - Running production ML workloads at scale
    - Want elastic resource management (auto-scaling)
    - Have Kubernetes infrastructure available
    - Need specialized compute for different tasks (CPUs, GPUs, memory)

!!! abstract "Use Local Container When"

    - Testing container-based pipelines before cloud deployment
    - Don't need parallel execution
    - Want simpler setup and debugging

!!! note "Use Local Executor When"

    - Development and experimentation
    - All tasks use the same environment
    - Want fastest possible development iteration

## Advanced: Complex Nested Workflows

!!! info "Nested Pipeline Support"

    Runnable supports deeply nested workflows with `Map` inside `Parallel` inside `Map` structures. Argo handles the complexity automatically - you just write simple Python pipeline code and Runnable generates the appropriate workflow DAGs.
