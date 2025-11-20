# Pipeline Execution Overview

Configure how multi-step workflows are orchestrated and executed across different environments.

## Available Pipeline Executors

| Executor | Use Case | Environment | Execution Model |
|----------|----------|-------------|-----------------|
| [Local](local.md) | Development | Local machine | **Sequential + Conditional Parallel*** |
| [Local Container](local-container.md) | Isolated development | Docker containers** | **Sequential + Conditional Parallel*** |
| [Argo Workflows](argo.md) | Production | Kubernetes** | **Parallel + Sequential** (full orchestration) |

*_Parallel execution requires `enable_parallel: true` and compatible run log store_
**_Container environments easily match local setup (just build from project root!)_

## Execution Models

### Local Execution Models

**Local** and **Local Container** executors support both sequential and parallel execution:

#### Sequential Execution (Default)
- ✅ **Fast startup**: No orchestration overhead
- ✅ **Simple debugging**: Linear execution, easy to trace
- ✅ **Resource efficient**: Single process, minimal memory usage
- ✅ **Universal compatibility**: Works with all run log stores

#### Conditional Parallel Execution
- ✅ **Optional parallelization**: Enable with `enable_parallel: true`
- ✅ **Automatic fallback**: Falls back to sequential if run log store doesn't support parallel writes
- ✅ **Local multiprocessing**: Uses your machine's multiple cores
- ⚠️ **Run log store dependency**: Requires `chunked-fs` or compatible run log store

**Best for**: Development, debugging, small-to-medium workflows, single-machine execution

### Parallel Execution (Orchestrated Executors)

**Argo Workflows** supports both sequential and parallel execution:

- ✅ **True parallelization**: Independent tasks run simultaneously
- ✅ **Complex workflows**: DAG-based execution with dependencies
- ✅ **Scalability**: Distributed across multiple nodes/pods
- ✅ **Production features**: Retry logic, monitoring, resource management
- ⚠️ **Higher overhead**: Kubernetes orchestration complexity

**Best for**: Production workflows, parallel processing, complex dependencies

## Key Concepts

**Pipeline executors** handle:

- **Workflow orchestration**: Managing step dependencies and execution order
- **Cross-step data flow**: Passing parameters and artifacts between tasks
- **Environment management**: Ensuring each step runs in the correct context
- **Failure handling**: Managing retries, error propagation, and cleanup

## Quick Start

```python
from runnable import Pipeline, PythonTask

pipeline = Pipeline(steps=[
    PythonTask(function=step1, name="process"),
    PythonTask(function=step2, name="analyze")
])

# Local execution (default)
pipeline.execute()

# Container execution
pipeline.execute(config="container.yaml")

# Production orchestration
pipeline.execute(config="argo.yaml")
```

## When to Use Pipeline Execution

Choose pipeline execution when you need:

- **Multi-step workflows** with dependencies between tasks
- **Cross-step data passing** via parameters or catalog

**For Sequential Workflows**:
- Use [Local](local.md) or [Local Container](local-container.md) for development and simple pipelines

**For Parallel Workflows**:
- Use [Argo Workflows](argo.md) for parallel branches, map-reduce patterns, and complex orchestration
- **Production deployment** with monitoring and failure handling

## When to Use Job Execution Instead

For single task execution, consider [Job Execution](../job-executors/overview.md):

- **Single functions** without workflow dependencies
- **Independent tasks** that don't share data
- **Simple execution** without orchestration complexity

---

**Related:** [Job Execution Overview](../job-executors/overview.md) | [Configuration Overview](../overview.md)
