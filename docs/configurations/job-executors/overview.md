# Job Execution Overview

Execute individual tasks across different environments with flexible runtime configurations.

## Jobs vs Pipelines

**Jobs**: Execute single tasks in isolation

```python
from runnable import PythonJob
from examples.common.functions import hello

def main():
    job = PythonJob(function=hello)
    job.execute()  # Single task execution
    return job

if __name__ == "__main__":
    main()
```

**Pipelines**: Orchestrate multiple connected tasks

```python
from runnable import Pipeline, PythonTask
from examples.common.functions import hello

def main():
    pipeline = Pipeline(steps=[
        PythonTask(function=hello, name="task1"),
        PythonTask(function=hello, name="task2")
    ])
    pipeline.execute()  # Multi-task workflow
    return pipeline

if __name__ == "__main__":
    main()
```

## Available Job Executors

| Executor | Use Case | Environment | Execution Model |
|----------|----------|-------------|-----------------|
| [Local](local.md) | Development | Local machine | **Direct execution** |
| [Local Container](local-container.md) | Isolated development | Docker containers | **Containerized execution** |
| [Kubernetes](kubernetes.md) | Production | Kubernetes cluster | **Distributed execution** |

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
uv run my_job.py

# Or inline for different environments
RUNNABLE_CONFIGURATION_FILE=production.yaml uv run my_job.py
```

**Alternative** (inline in code):
```python
job.execute(configuration_file="config.yaml")
```

!!! info "Examples Directory"

    Complete working examples are available in `examples/11-jobs/`. Each example includes both Python code and YAML configuration files you can run immediately.

## Choosing the Right Executor

### Development & Testing
- **[Local](local.md)**: Quick development, debugging, simple tasks
- **[Local Container](local-container.md)**: Isolated development, dependency consistency

### Production Deployment
- **[Kubernetes](kubernetes.md)**: Production scale, resource management, distributed execution

### When to Use Job Execution

Choose job execution when you need:

- **Single task execution** without workflow orchestration
- **Independent tasks** that don't share data with other steps
- **Simple execution** without complex dependencies

### When to Use Pipeline Execution Instead

For multi-task workflows, consider [Pipeline Execution](../executors/overview.md):

- **Multi-step workflows** with dependencies between tasks
- **Cross-step data passing** via parameters or catalog
- **Complex orchestration** with parallel branches or conditional logic

## Next Steps

1. **Start simple**: Begin with [Local](local.md) execution for development
2. **Add isolation**: Move to [Local Container](local-container.md) for consistent environments
3. **Scale up**: Deploy with [Kubernetes](kubernetes.md) for production workloads

---

!!! tip "Multi-Task Workflows"

    For orchestrating multiple connected tasks, see [Pipeline Execution](../executors/local.md) which provides workflow management and cross-step data passing.
