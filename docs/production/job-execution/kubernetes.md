# Kubernetes Job Execution

Execute jobs on Kubernetes clusters with production-grade resource management, persistence, and scalability.

!!! info "Installation Required"

    Kubernetes execution requires the optional Kubernetes dependency:
    ```bash
    pip install runnable[k8s]
    ```

## Why Use Kubernetes Execution?

!!! success "Production Benefits"

    **Enterprise-ready orchestration**: Production-scale job execution with Kubernetes

    - 🏗️ **Resource management**: CPU, memory, GPU limits and requests
    - 💾 **Persistent storage**: Shared data across job runs
    - 🔄 **Scalability**: Leverage multi-node cluster resources
    - 📊 **Monitoring**: Native Kubernetes observability and logging

!!! note "Trade-offs"

    - 🐳 **Infrastructure requirement**: Needs Kubernetes cluster setup
    - ⚙️ **Complexity**: More moving parts than local executors
    - 🚀 **Pod overhead**: ~10-30 seconds startup time

## Kubernetes Variants

Runnable provides two Kubernetes job executors for different cluster setups:

=== "Production Kubernetes"

    **Use for**: Real Kubernetes clusters with persistent storage

    ```yaml
    job-executor:
      type: k8s-job
      config:
        pvc_claim_name: "runnable-storage"  # REQUIRED
        jobSpec:
          template:
            spec:
              container:
                image: "my-project:v1.0"  # REQUIRED - your project image
    ```

=== "Minikube Development"

    **Use for**: Local Kubernetes development with minikube

    ```yaml
    job-executor:
      type: mini-k8s-job
      config:
        jobSpec:
          template:
            spec:
              container:
                image: "my-project:latest"  # REQUIRED - your project image
    ```

!!! tip "Container Setup Made Simple"

    Just build a Docker image from your project root and push to your registry:

    ```bash
    # Build from your project root
    docker build -t my-project:v1.0 .
    docker push your-registry.com/my-project:v1.0
    ```

!!! info "Standard Kubernetes Job Specification"

    The `jobSpec` configuration follows the standard [Kubernetes Job API specification](https://kubernetes.io/docs/reference/kubernetes-api/workload-resources/job-v1/). Runnable uses native Kubernetes Job configuration - you can reference the official Kubernetes documentation for all available fields and options.

## Getting Started

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

=== "k8s-config.yaml"

    ```yaml
    job-executor:
      type: k8s-job
      config:
        pvc_claim_name: "runnable-storage"
        jobSpec:
          template:
            spec:
              container:
                image: "my-project:v1.0"
    ```

**Run the Kubernetes job:**
```bash
# Push your image to registry first
docker push your-registry.com/my-project:v1.0

# Run with Kubernetes executor
RUNNABLE_CONFIGURATION_FILE=k8s-config.yaml uv run job.py
```

!!! info "Pod Execution"

    Your job runs in a Kubernetes pod with isolated resources and access to cluster storage.

## Configuration Reference

### Production Kubernetes (k8s-job)

```yaml
job-executor:
  type: k8s-job
  config:
    pvc_claim_name: "runnable-storage"  # Required: PVC for data persistence
    namespace: "default"               # Optional: Kubernetes namespace
    config_path: null                  # Optional: Path to kubeconfig file
    mock: false                        # Optional: Skip execution for testing
    jobSpec:                          # Required: Kubernetes Job specification
      template:
        spec:
          container:
            image: "my-project:v1.0"   # Required: Docker image to use
            env:                       # Optional: Environment variables
              - name: "LOG_LEVEL"
                value: "INFO"
            resources:                 # Optional: Resource limits
              limits:
                cpu: "2"
                memory: "4Gi"
              requests:
                cpu: "1"
                memory: "2Gi"
```

### Minikube Development (mini-k8s-job)

```yaml
job-executor:
  type: mini-k8s-job
  config:
    namespace: "default"               # Optional: Kubernetes namespace
    mock: false                        # Optional: Skip execution for testing
    jobSpec:                          # Required: Kubernetes Job specification
      template:
        spec:
          container:
            image: "my-project:latest" # Required: Docker image to use
```

### Common Configuration Options

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pvc_claim_name` | str | **Yes (k8s-job)** | PVC name for data persistence |
| `jobSpec.template.spec.container.image` | str | **Yes** | Docker image for job execution |
| `namespace` | str | No | Kubernetes namespace (default: "default") |
| `config_path` | str | No | Path to kubeconfig file |
| `mock` | bool | No | Simulate execution without creating job |

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

=== "k8s-job.yaml"

    ```yaml title="examples/11-jobs/k8s-job.yaml (simplified)"
    job-executor:
      type: "k8s-job"
      config:
        pvc_claim_name: runnable-storage
        namespace: default
        jobSpec:
          template:
            spec:
              container:
                image: your-registry.com/your-project:latest
    ```

**Run the example:**
```bash
# Push your image to the registry
docker push your-registry.com/your-project:latest

# Run with Kubernetes executor
RUNNABLE_CONFIGURATION_FILE=examples/11-jobs/k8s-job.yaml uv run examples/11-jobs/python_tasks.py
```

## Prerequisites

### Kubernetes Setup

**Verify cluster access:**
```bash
kubectl cluster-info
kubectl get nodes
```

**Create PVC for production (k8s-job only):**
```yaml title="runnable-pvc.yaml"
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: runnable-storage
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
```

```bash
kubectl apply -f runnable-pvc.yaml
```

## Troubleshooting

### Common Issues

!!! warning "Job Stuck in Pending"

    **Problem**: Pod doesn't start

    **Solution**:
    ```bash
    # Check pod status and events
    kubectl get pods -l job-name=<run-id>
    kubectl describe pod <pod-name>
    ```

!!! warning "Image Pull Errors"

    **Problem**: Cannot pull container image

    **Solution**:
    ```bash
    # Verify image exists and is accessible
    docker pull <your-image>

    # Check image pull secrets for private registries
    kubectl get secrets
    ```

### Debug Mode

Test job configuration without creating actual pods:

```yaml
job-executor:
  type: k8s-job
  config:
    mock: true  # Logs job spec without creating actual job
    pvc_claim_name: "runnable-storage"
    jobSpec:
      template:
        spec:
          container:
            image: "my-project:latest"
```

## Advanced Features

### Job Scheduling

Schedule jobs to run automatically using Kubernetes CronJobs:

```yaml
job-executor:
  type: k8s-job
  config:
    schedule: "0 2 * * *"  # Run daily at 2 AM
    pvc_claim_name: "runnable-storage"
    jobSpec:
      template:
        spec:
          container:
            image: "my-project:v1.0"
```

**Learn more:** [Kubernetes Job Scheduling Guide](k8s-scheduling.md)

## When to Use Other Executors

Consider alternatives when you need:

!!! abstract "Local Development"

    **[Local](local.md)**: For simple development without container overhead

    **[Local Container](local-container.md)**: For containerized development without Kubernetes

!!! note "Simpler Infrastructure"

    **[Local Container](local-container.md)**: When you need containers but not full Kubernetes orchestration

---

**Related:** [K8s Job Scheduling](k8s-scheduling.md) | [Pipeline Argo Workflows](../pipeline-execution/argo.md) | [All Job Executors](overview.md)
