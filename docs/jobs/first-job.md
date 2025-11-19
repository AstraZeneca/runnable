# Your First Job

Let's start with the simplest possible Job: running a function once.

## The Basics

A Job wraps your function to provide execution tracking, logging, and output storage.

```python
from examples.common.functions import hello
from runnable import PythonJob

def main():
    job = PythonJob(function=hello)
    job.execute()
    return job

if __name__ == "__main__":
    main()
```

??? example "See complete runnable code"
    ```python title="examples/11-jobs/python_tasks.py"
    --8<-- "examples/11-jobs/python_tasks.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/11-jobs/python_tasks.py
    ```

## What Happens When You Run It

**1. Rich Context Display**
```
🏃 Lets go!!
Working with context:
JobContext(
    run_id='null-panini-0628',
    catalog=FileSystemCatalog(catalog_location='.catalog'),
    job_executor=LocalJobExecutor(),
    ...
)
```

**2. Function Execution**
```
Hello World!
WARNING:This is a warning message.
```

**3. Automatic Storage**
- Execution logs captured in `.catalog/{run-id}/`
- Unique run ID generated for tracking
- All output preserved automatically

**4. Summary Report**
```json
{
    "Available parameters": [],
    "Output parameters": [],
    "status": "SUCCESS"
}
```

## Generated Run IDs

Each execution gets a unique, memorable run ID:
- `null-panini-0628`
- `minty-brattain-0628`
- `feasible-booth-0628`

These organize your catalog and make runs easy to find.

## Job Types at a Glance

| Job Type | Purpose | Example Use |
|----------|---------|-------------|
| **PythonJob** | Execute Python functions | Data analysis, calculations |
| **ShellJob** | Run shell commands | File processing, system ops |
| **NotebookJob** | Execute Jupyter notebooks | Interactive analysis, reports |

## What's Next?

Now that you can run basic Jobs:

- **[Working with Data](working-with-data.md)** - Store and return function outputs
- **[Parameters & Environment](parameters.md)** - Configure Jobs without code changes
- **[File Storage](file-storage.md)** - Automatically archive Job outputs

Ready to return data from your Jobs? Continue to **[Working with Data](working-with-data.md)**!
