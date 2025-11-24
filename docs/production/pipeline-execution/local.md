# Local Pipeline Execution

Execute pipelines in your local environment with optional parallel processing - perfect for development and testing.

All pipeline steps execute in the same local environment where they were triggered.

## Features

- ✅ **Comfortable development**: Direct access to your local environment
- ✅ **Optional parallelization**: Enable parallel execution for `parallel` and `map` nodes
- ✅ **Automatic fallback**: Gracefully falls back to sequential if run log store doesn't support parallel writes
- ⚠️ **Local resource constraints**: Scalability limited by your machine's resources
- ❌ **Single environment**: All steps share the same compute environment



## Quick Start

### Sequential Execution (Default)

```python
from runnable import Pipeline, PythonTask

def step_one():
    print("Step 1 running...")
    return "data_from_step_1"

def step_two():
    print("Step 2 running...")
    return "final_result"

def main():
    pipeline = Pipeline(steps=[
        PythonTask(function=step_one, name="step1"),
        PythonTask(function=step_two, name="step2")
    ])

    # Sequential execution (default)
    pipeline.execute()

if __name__ == "__main__":
    main()
```

### Parallel Execution

Enable parallel processing for `parallel` and `map` nodes:

=== "pipeline.py"

    ```python
    from runnable import Pipeline, PythonTask, Parallel

    def process_data(item):
        print(f"Processing {item}")
        return f"processed_{item}"

    def main():
        # Create parallel branches
        parallel_node = Parallel(
            name="process_parallel",
            branches={
                "branch_1": [PythonTask(function=process_data, name="task1")],
                "branch_2": [PythonTask(function=process_data, name="task2")],
                "branch_3": [PythonTask(function=process_data, name="task3")]
            }
        )

        pipeline = Pipeline(steps=[parallel_node])

        # Enable parallel execution
        pipeline.execute(configuration_file="parallel_config.yaml")

        return pipeline

    if __name__ == "__main__":
        main()
    ```

=== "parallel_config.yaml"

    ```yaml
    pipeline-executor:
      type: local
      config:
        enable_parallel: true

    # Parallel-compatible run log store
    run-log-store:
      type: chunked-fs

    catalog:
      type: file-system
    ```


**Run the pipeline:**
```bash
uv run pipeline.py
```

!!! success "Automatic Fallback"

    If you use a run log store that doesn't support parallel writes (like `file-system`), the executor automatically falls back to sequential execution with a helpful warning.

## Parallel Execution Requirements

For parallel execution to work, you need:

1. **Enable parallel**: Set `enable_parallel: true` in pipeline executor config
2. **Compatible run log store**: Use `chunked-fs` (file-system doesn't support parallel writes)

| Run Log Store | Parallel Support | Use Case |
|---------------|------------------|----------|
| `file-system` | ❌ Sequential only | Simple development |
| `chunked-fs` | ✅ **Parallel ready** | **Parallel local execution** |

## Configuration Reference

```yaml
pipeline-executor:
  type: local
  config:
    enable_parallel: true
```

## When to Use Local Execution

!!! success "Perfect for Development"

    - **Quick experimentation** with immediate feedback
    - **Debugging** with direct access to your environment
    - **Small to medium datasets** that fit in local memory
    - **Iterative development** with fast execution cycles

### Sequential vs Parallel

**Use sequential (default) when:**

- Simple linear pipelines
- Steps depend heavily on each other
- You want minimal overhead and complexity

**Use parallel when:**

- You have independent `parallel` or `map` branches
- Your machine has multiple cores to utilize
- You're testing parallel patterns before deploying to production

!!! tip "Development Foundation"

    All the conceptual examples use `local` executors for simplicity. Start here, then move to production executors when ready.
