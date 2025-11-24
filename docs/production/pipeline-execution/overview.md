# Pipeline Execution Overview

Configure how multi-step workflows are orchestrated and executed across different environments.

## Available Pipeline Executors

| Executor | Use Case | Environment | Execution Model |
|----------|----------|-------------|-----------------|
| [Local](local.md) | Development | Local machine | **Sequential + Conditional Parallel*** |
| [Local Container](local-container.md) | Isolated development | Docker containers** | **Sequential + Conditional Parallel*** |
| [Argo Workflows](argo.md) | Production | Kubernetes** | **Parallel + Sequential** (full orchestration) |
| [Mocked](mocked.md) | Testing & validation | Local machine | **Simulation** (no actual execution) |

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

### Production Execution (Orchestrated)

**Argo Workflows** supports both sequential and parallel execution:

- ✅ **True parallelization**: Independent tasks run simultaneously
- ✅ **Complex workflows**: DAG-based execution with dependencies
- ✅ **Scalability**: Distributed across multiple nodes/pods
- ✅ **Production features**: Retry logic, monitoring, resource management
- ⚠️ **Higher overhead**: Kubernetes orchestration complexity

**Best for**: Production workflows, parallel processing, complex dependencies

### Testing and Validation (Local)

**Mocked Executor** provides pipeline validation without execution:

- ✅ **Fast validation**: Check pipeline structure without running tasks
- ✅ **Configuration testing**: Validate configs before deployment
- ✅ **Development workflow**: Test pipeline logic without side effects
- ✅ **Local only**: Runs on your machine for testing purposes

**Best for**: Pipeline validation, testing configurations, development workflows

## Key Concepts

**Pipeline executors** handle:

- **Workflow orchestration**: Managing step dependencies and execution order
- **Cross-step data flow**: Passing parameters and artifacts between tasks
- **Environment management**: Ensuring each step runs in the correct context
- **Failure handling**: Managing retries, error propagation, and cleanup

## Quick Start

```python
from runnable import Pipeline, PythonTask

def step1():
    # Your processing logic
    return "processed_data"

def step2(processed_data):
    # Your analysis logic
    return "analysis_result"

def main():
    pipeline = Pipeline(steps=[
        PythonTask(function=step1, name="process"),
        PythonTask(function=step2, name="analyze")
    ])

    # Environment determines executor via RUNNABLE_CONFIGURATION_FILE
    pipeline.execute()
    return pipeline

if __name__ == "__main__":
    main()
```

**Run with different executors:**

```bash
# Local execution (default)
uv run my_pipeline.py

# Container execution
export RUNNABLE_CONFIGURATION_FILE=local-container.yaml
uv run my_pipeline.py

# Production orchestration
export RUNNABLE_CONFIGURATION_FILE=argo.yaml
uv run my_pipeline.py
```

## Custom Pipeline Executors

**Need to deploy to your unique infrastructure?** Runnable's plugin architecture makes it simple to build custom pipeline executors for any orchestration platform.

!!! success "No Vendor Lock-in"

    **Your infrastructure, your way**: Deploy pipelines on Argo Workflows, Apache Airflow, Azure Data Factory, or any custom orchestration platform.

    - 🔌 **Cloud orchestrators**: Argo Workflows, Apache Airflow, Prefect, Azure Data Factory
    - 🏢 **HPC systems**: SLURM, PBS, custom job schedulers
    - 🎯 **Container platforms**: Kubernetes, Docker Swarm, Nomad
    - 🔐 **Enterprise platforms**: Custom workflow engines, proprietary orchestrators

### Building Custom Pipeline Executors

Learn how to create production-ready custom pipeline executors:

**[📖 Custom Pipeline Executors Development Guide](custom-pipeline-executors.md)**

The guide provides:

- **Complete stubbed implementation** showing integration patterns
- **Node-by-node vs full DAG transpilation** execution models
- **YAML to Pydantic configuration mapping** with validation
- **Testing workflow** with mock modes for safe development

!!! example "Quick Example"

    Create a custom pipeline executor in just 3 steps:

    1. **Implement key methods** by extending `GenericPipelineExecutor`
    2. **Register via entry point** in your `pyproject.toml`
    3. **Configure via YAML** for seamless integration

    ```python
    from extensions.pipeline_executor import GenericPipelineExecutor

    class MyPlatformExecutor(GenericPipelineExecutor):
        service_name: str = "my-platform"

        def trigger_node_execution(self, node, map_variable=None):
            # Your orchestration platform integration here
            pass
    ```

**Ready to build?** See the [development guide](custom-pipeline-executors.md) for complete patterns and examples.

## Choosing the Right Executor

### Development & Testing
- **[Local](local.md)**: Quick development, debugging, small workflows
- **[Local Container](local-container.md)**: Isolated development, dependency consistency
- **[Mocked](mocked.md)**: Pipeline validation, configuration testing

### Production Deployment
- **[Argo Workflows](argo.md)**: Production orchestration, parallel processing, complex workflows

### When to Use Pipeline Execution

Choose pipeline execution when you need:

- **Multi-step workflows** with dependencies between tasks
- **Cross-step data passing** via parameters or catalog
- **Complex orchestration** with parallel branches or conditional logic

### When to Use Job Execution Instead

For single task execution, consider [Job Execution](../job-execution/overview.md):

- **Single functions** without workflow dependencies
- **Independent tasks** that don't share data
- **Simple execution** without orchestration complexity

---

**Related:** [Job Execution Overview](../job-execution/overview.md) | [Configuration Overview](../overview.md)
