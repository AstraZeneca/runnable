# Kubernetes Job Execution

Execute jobs on Kubernetes clusters with full production capabilities including resource management, persistence, and scalability.

## When to Use

- **Production deployment**: Scalable, managed execution environment
- **Resource management**: CPU, memory, GPU limits and requests
- **Persistent storage**: Shared data across job runs
- **Multi-node clusters**: Leverage cluster resources and scheduling

## Kubernetes Variants

Runnable provides three Kubernetes job executors for different cluster setups:

=== "Production K8s"

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
                image: "recommendation-engine:v2.1.3"  # REQUIRED - your project image
    ```

    !!! tip "Environment Parity Made Simple"

        Your Kubernetes container just needs to match your local environment. **Easy setup**:

        ```bash
        # Build from your project root
        docker build -t my-project:v1.0 .
        docker push your-registry.com/my-project:v1.0
        ```

        Container automatically inherits your local setup! ✨

=== "Minikube"

    **Use for**: Local Kubernetes development with minikube

    ```yaml
    job-executor:
      type: mini-k8s-job
      config:
        jobSpec:
          template:
            spec:
              container:
                image: "data-pipeline:minikube"  # REQUIRED - your project image for minikube
    ```

    !!! tip "Environment Parity Made Simple"

        Container setup is easy - just build from your project root!

        ```bash
        docker build -t my-project:minikube .
        ```

!!! info "Standard Kubernetes Job Specification"

    The `jobSpec` configuration follows the standard [Kubernetes Job API specification](https://kubernetes.io/docs/reference/kubernetes-api/workload-resources/job-v1/). Runnable doesn't invent a new schema - it uses native Kubernetes Job configuration, so you can reference the official Kubernetes documentation for all available fields and options.

## Essential Configuration

### Production Kubernetes

**Required fields**:

```yaml
job-executor:
  type: k8s-job
  config:
    pvc_claim_name: "runnable-storage"  # PVC for data persistence
    jobSpec:
      template:
        spec:
          container:
            image: "harbor.company.com/ml-team/sentiment-analysis:v1.4.2"
```

### Minikube Development

**Required fields**:

```yaml
job-executor:
  type: mini-k8s-job
  config:
    jobSpec:
      template:
        spec:
          container:
            image: "analytics-pipeline:minikube-v1.0"
```

## Common Customizations

### Namespace and Configuration

```yaml
job-executor:
  type: k8s-job
  config:
    namespace: "ml-workloads"          # Default: "default"
    config_path: "/path/to/kubeconfig" # Optional: custom kubeconfig
    pvc_claim_name: "runnable-storage"
    jobSpec:
      template:
        spec:
          container:
            image: "fraud-detection-service:v1.0.0"
```

### Resource Limits

```yaml
job-executor:
  type: k8s-job
  config:
    pvc_claim_name: "runnable-storage"
    jobSpec:
      template:
        spec:
          container:
            image: "customer-segmentation:v3.2.1"
            resources:
              limits:
                cpu: "4"
                memory: "8Gi"
                nvidia.com/gpu: "1"
              requests:
                cpu: "2"
                memory: "4Gi"
```

### Environment Variables

```yaml
job-executor:
  type: k8s-job
  config:
    pvc_claim_name: "runnable-storage"
    jobSpec:
      template:
        spec:
          container:
            image: "time-series-forecaster:latest"
            env:
              - name: "DATABASE_URL"
                value: "postgresql://db:5432/prod"
              - name: "LOG_LEVEL"
                value: "INFO"
```

## Advanced Configuration

### Job Timeouts and Retry Policy

```yaml
job-executor:
  type: k8s-job
  config:
    pvc_claim_name: "runnable-storage"
    jobSpec:
      activeDeadlineSeconds: 3600  # 1 hour timeout
      backoffLimit: 3              # Retry up to 3 times
      template:
        spec:
          restartPolicy: "Never"   # Don't restart failed pods
          container:
            image: "time-series-forecaster:latest"
```

### Node Selection and Tolerations

??? example "Advanced scheduling"

    ```yaml
    job-executor:
      type: k8s-job
      config:
        pvc_claim_name: "runnable-storage"
        jobSpec:
          template:
            spec:
              nodeSelector:
                node-type: "gpu-node"
              tolerations:
                - key: "gpu"
                  operator: "Equal"
                  value: "true"
                  effect: "NoSchedule"
              container:
                image: "deep-learning-trainer:v2.0-gpu"
                resources:
                  limits:
                    nvidia.com/gpu: "2"
    ```

### Custom Volumes

??? example "Additional storage"

    ```yaml
    job-executor:
      type: k8s-job
      config:
        pvc_claim_name: "runnable-storage"
        jobSpec:
          template:
            spec:
              volumes:
                - name: "data-cache"
                  hostPath:
                    path: "/mnt/fast-ssd"
              container:
                image: "time-series-forecaster:latest"
                volumeMounts:
                  - name: "data-cache"
                    mountPath: "/cache"
    ```

## Complete Examples

=== "Production Setup"

    ```python title="examples/11-jobs/python_tasks.py"
    --8<-- "examples/11-jobs/python_tasks.py"
    ```

    ```yaml title="examples/11-jobs/k8s-job.yaml"
    --8<-- "examples/11-jobs/k8s-job.yaml"
    ```

    ```bash
    # Recommended: Environment variable approach
    export RUNNABLE_CONFIGURATION_FILE=examples/11-jobs/k8s-job.yaml
    uv run examples/11-jobs/python_tasks.py

    # Alternative: Inline config flag
    uv run examples/11-jobs/python_tasks.py --config examples/11-jobs/k8s-job.yaml
    ```

=== "Minikube Setup"

    ```yaml title="examples/11-jobs/mini-k8s-job.yaml"
    --8<-- "examples/11-jobs/mini-k8s-job.yaml"
    ```

    ```bash
    # First setup minikube volumes
    minikube mount $HOME/workspace/runnable/.run_log_store:/volume/run_logs &
    minikube mount $HOME/workspace/runnable/.catalog:/volume/catalog &

    # Then run the job (recommended approach)
    export RUNNABLE_CONFIGURATION_FILE=examples/11-jobs/mini-k8s-job.yaml
    uv run examples/11-jobs/python_tasks.py

    # Alternative: inline config
    uv run examples/11-jobs/python_tasks.py --config examples/11-jobs/mini-k8s-job.yaml
    ```

## Prerequisites

### Kubernetes Access

```bash
# Verify cluster access
kubectl cluster-info

# Check available nodes
kubectl get nodes

# Verify namespace exists (or create it)
kubectl create namespace ml-workloads
```

### Persistent Volume Claim (Production)

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

### Service Account Permissions

```yaml title="runnable-rbac.yaml"
apiVersion: v1
kind: ServiceAccount
metadata:
  name: runnable-executor
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: job-executor
rules:
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["create", "get", "list", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: runnable-job-executor
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: job-executor
subjects:
- kind: ServiceAccount
  name: runnable-executor
```

## Configuration Reference

### Essential Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `jobSpec.template.spec.container.image` | str | **Yes** | Docker image for job execution |
| `pvc_claim_name` | str | **Yes (prod)** | PVC name for data persistence |

### Common Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `namespace` | str | `"default"` | Kubernetes namespace |
| `config_path` | str | `None` | Path to kubeconfig file |
| `mock` | bool | `false` | Simulate execution without creating job |

### Job Specification

The `jobSpec` follows the standard [Kubernetes Job API](https://kubernetes.io/docs/reference/kubernetes-api/workload-resources/job-v1/) with these commonly used fields:

- `activeDeadlineSeconds`: Job timeout in seconds (default: 7200)
- `backoffLimit`: Number of retries (default: 6)
- `template.spec.restartPolicy`: Pod restart policy (default: "Never")
- `template.spec.container`: Container specification
- `template.spec.nodeSelector`: Node selection constraints
- `template.spec.tolerations`: Node taint tolerations

## Volume Management

Runnable automatically manages volumes for:

- **Run logs**: Job execution metadata and logs
- **Catalog**: Data artifacts and intermediate results
- **Secrets**: Environment variables and configuration

Volume mounting strategy:

- **Production**: Uses PVC for shared, persistent storage
- **Minikube**: Uses host path volumes for local development
- **Custom**: Define additional volumes in `jobSpec.template.spec.volumes`

## Troubleshooting

### Common Issues

**Job stuck in Pending**:

```bash
# Check pod status
kubectl get pods -l job-name=<run-id>

# Describe pod for events
kubectl describe pod <pod-name>
```

**Image pull errors**:

```bash
# Check image exists and is accessible
docker pull <your-image>

# Verify image pull secrets if using private registry
kubectl get secrets
```

**Resource constraints**:

```bash
# Check node resources
kubectl describe nodes

# Check resource quotas
kubectl describe quota -n <namespace>
```

### Debug Mode

Enable mock mode for testing:

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
            image: "time-series-forecaster:latest"
```

## Performance Considerations

- **Pod startup time**: ~10-30 seconds depending on image size
- **Resource allocation**: Set appropriate requests/limits
- **Storage I/O**: PVC performance varies by storage class
- **Network**: Consider cluster networking for external dependencies

## When to Use Other Executors

Consider alternatives when you need:

- **Local development**: → [Local](local.md) or [Local Container](local-container.md)
- **Simple containerization**: → [Local Container](local-container.md)

---

**Related:** [Pipeline Argo Workflows](../executors/argo.md) | [All Job Executors](overview.md)
