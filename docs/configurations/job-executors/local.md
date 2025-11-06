# Local Job Execution

Execute jobs directly on your local machine without containerization or orchestration.

## When to Use

- **Quick development**: Rapid iteration and debugging
- **Simple tasks**: No special environment requirements
- **Getting started**: Learning Runnable without complexity
- **Local resources**: Direct access to local files and environment

## Quick Start

```python
from runnable import PythonJob
from examples.common.functions import hello

job = PythonJob(function=hello)
job.execute()  # Uses local executor by default
```

## Configuration

### No Configuration Required

Local execution is the **default** - no configuration file needed:

```python
job.execute()  # Automatically uses local executor
```

### Optional Configuration

If you want to be explicit or add options:

```yaml title="local-job.yaml"
job-executor:
  type: local
  config:
    mock: false  # Set to true for testing workflow logic
```

```python
job.execute(config="local-job.yaml")
```

## Configuration Options

### Essential Options

**No required fields** - all configuration is optional.

### Available Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mock` | bool | `false` | Skip actual execution, simulate success |

??? example "Mock execution for testing"

    ```yaml
    job-executor:
      type: local
      config:
        mock: true  # Simulate execution without running code
    ```

    Useful for:

    - Testing workflow logic
    - Validating job configuration
    - Dry-run scenarios

## Complete Example

=== "Python Code"

    ```python title="examples/11-jobs/python_tasks.py"
    --8<-- "examples/11-jobs/python_tasks.py"
    ```

=== "Run It"

    ```bash
    # No config needed - runs locally by default
    uv run examples/11-jobs/python_tasks.py
    ```

=== "With Explicit Config"

    ```yaml title="local-config.yaml"
    job-executor:
      type: local
      config:
        mock: false
    ```

    ```bash
    # Recommended: Environment variable approach
    export RUNNABLE_CONFIGURATION_FILE=local-config.yaml
    uv run examples/11-jobs/python_tasks.py

    # Alternative: Inline config flag
    uv run examples/11-jobs/python_tasks.py --config local-config.yaml
    ```

## Environment Variables

Local jobs have direct access to your environment variables:

```python
import os

def my_function():
    db_url = os.getenv('DATABASE_URL', 'sqlite://default.db')
    return f"Using database: {db_url}"

job = PythonJob(function=my_function)
job.execute()
```

## File Access

Direct access to local filesystem:

```python
def process_file():
    with open('data/input.csv', 'r') as f:
        # Process file directly
        return len(f.readlines())

job = PythonJob(function=process_file)
job.execute()
```

## Performance Characteristics

- **Fast startup**: No container overhead
- **Direct I/O**: No network or mount overhead
- **Full system access**: All local resources available
- **No isolation**: Shares environment with your system

## When to Graduate

Consider upgrading to other executors when you need:

- **Environment isolation**: → [Local Container](local-container.md)
- **Production deployment**: → [Kubernetes](kubernetes.md)
- **Consistent environments**: → [Local Container](local-container.md)
- **Resource limits**: → [Kubernetes](kubernetes.md)

---

**Related:** [Pipeline Local Execution](../executors/local.md) | [All Job Executors](overview.md)
