# Job Execution Overview

Execute individual tasks across different environments with flexible runtime configurations.

## Jobs vs Pipelines

**Jobs**: Execute single tasks in isolation

```python
from runnable import PythonJob
from examples.common.functions import hello

job = PythonJob(function=hello)
job.execute()  # Single task execution
```

**Pipelines**: Orchestrate multiple connected tasks

```python
from runnable import Pipeline, PythonTask
from examples.common.functions import hello

pipeline = Pipeline(steps=[
    PythonTask(function=hello, name="task1"),
    PythonTask(function=hello, name="task2")
])
pipeline.execute()  # Multi-task workflow
```

## Choose Your Environment

| Environment | Use Case | Required Config | Examples |
|-------------|----------|----------------|----------|
| [Local](local.md) | Quick development | None | `job.execute()` |
| [Local Container](local-container.md) | Isolated execution | Docker image | Container isolation |
| [Kubernetes](kubernetes.md) | Production scale | K8s job spec | Resource limits, PVC |

## Configuration Pattern

All job executors use this configuration pattern:

```yaml title="config.yaml"
job-executor:
  type: "[executor-type]"
  config:
    # executor-specific options
```

**Recommended Usage** (via environment variable):
```bash
# Keep configuration separate from code
export RUNNABLE_CONFIGURATION_FILE=config.yaml
python my_job.py

# Or inline for different environments
RUNNABLE_CONFIGURATION_FILE=production.yaml python my_job.py
```

**Alternative** (inline in code):
```python
job.execute(config="config.yaml")
```

!!! info "Examples Directory"

    Complete working examples are available in `examples/11-jobs/`. Each example includes both Python code and YAML configuration files you can run immediately.

## Quick Decision Tree

- **Just developing locally?** → [Local](local.md)
- **Need environment isolation?** → [Local Container](local-container.md)
- **Deploying to production?** → [Kubernetes](kubernetes.md)

## Next Steps

1. **Start simple**: Begin with [Local](local.md) execution for development
2. **Add isolation**: Move to [Local Container](local-container.md) for consistent environments
3. **Scale up**: Deploy with [Kubernetes](kubernetes.md) for production workloads

---

!!! tip "Multi-Task Workflows"

    For orchestrating multiple connected tasks, see [Pipeline Execution](../executors/local.md) which provides workflow management and cross-step data passing.
