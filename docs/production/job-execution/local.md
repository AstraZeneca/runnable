# Local Job Execution

Execute jobs directly on your local machine without containerization or orchestration - perfect for development, experimentation, and simple task execution.

## Why Use Local Execution?

!!! success "Development Benefits"

    **Fast iteration**: No container overhead or deployment complexity

    - ⚡ **Instant execution**: Direct code execution on your machine
    - 🛠️ **Easy debugging**: Full access to development tools and debuggers
    - 📁 **Direct file access**: No mount points or volume mapping needed
    - 🔄 **Quick changes**: Edit and run immediately

!!! note "Trade-offs"

    - 🏠 **Local only**: Runs on your development machine
    - 🔗 **No isolation**: Shares your system environment
    - 💻 **Single machine**: Limited to local compute resources

## Getting Started

### Simple Example

=== "job.py"

    ```python
    from runnable import PythonJob
    from examples.common.functions import hello

    def main():
        job = PythonJob(function=hello)
        job.execute()  # Uses local executor by default
        return job

    if __name__ == "__main__":
        main()
    ```

=== "Run It"

    ```bash
    # No configuration needed - local is the default
    uv run job.py
    ```

**Result**: Your job runs directly on your machine using the current Python environment.

## Configuration

### No Configuration Required

Local execution is the **default** - no configuration file needed:

```python
def main():
    job = PythonJob(function=my_function)
    job.execute()  # Automatically uses local executor
    return job
```

### Optional Configuration

When you need explicit configuration or testing options:

```yaml title="local-job.yaml"
job-executor:
  type: local
  config:
    mock: false  # Set to true for testing workflow logic
```

**Recommended Usage** (via environment variable):
```bash
export RUNNABLE_CONFIGURATION_FILE=local-job.yaml
uv run my_job.py
```

**Alternative** (inline in code):
```python
def main():
    job = PythonJob(function=my_function)
    job.execute(configuration_file="local-job.yaml")
    return job
```

## Configuration Reference

```yaml
job-executor:
  type: local
  config:
    mock: false  # Optional: Skip actual execution for testing
```

### Configuration Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mock` | bool | `false` | Skip actual execution, simulate success for testing |

!!! example "Mock Execution for Testing"

    Test your job workflow without running actual code:

    ```yaml
    job-executor:
      type: local
      config:
        mock: true  # Simulate execution without running code
    ```

    **Use cases:**

    - Testing job workflow logic
    - Validating job configuration
    - Dry-run scenarios during development

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

=== "With Configuration"

    ```yaml title="local-config.yaml"
    job-executor:
      type: local
      config:
        mock: false
    ```

**Run the example:**
```bash
# No config needed - runs locally by default
uv run examples/11-jobs/python_tasks.py

# Or with explicit configuration
RUNNABLE_CONFIGURATION_FILE=local-config.yaml uv run examples/11-jobs/python_tasks.py
```

## Environment Variables

Local jobs have direct access to your environment variables:

```python
import os
from runnable import PythonJob

def access_environment():
    db_url = os.getenv('DATABASE_URL', 'sqlite://default.db')
    return f"Using database: {db_url}"

def main():
    job = PythonJob(function=access_environment)
    job.execute()
    return job
```

## File Access

Direct access to local filesystem without mount points:

```python
from runnable import PythonJob

def process_file():
    with open('data/input.csv', 'r') as f:
        # Process file directly from local filesystem
        return len(f.readlines())

def main():
    job = PythonJob(function=process_file)
    job.execute()
    return job
```

## Performance Characteristics

!!! success "Local Execution Benefits"

    - **Fast startup**: No container or orchestration overhead
    - **Direct I/O**: No network or mount overhead for file access
    - **Full system access**: All local resources and tools available
    - **Instant debugging**: Use your IDE debugger directly

!!! warning "Limitations"

    - **No isolation**: Shares environment with your system
    - **Single machine**: Limited to local compute resources
    - **Environment drift**: Different behavior across development machines

## When to Upgrade

Consider other executors when you need:

!!! abstract "Environment Consistency"

    **[Local Container](local-container.md)**: For isolated, reproducible environments

!!! success "Production Deployment"

    **[Kubernetes](kubernetes.md)**: For production workloads with resource management

---

**Related:** [Pipeline Local Execution](../pipeline-execution/local.md) | [All Job Executors](overview.md)
