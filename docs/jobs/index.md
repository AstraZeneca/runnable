# Jobs: Run Functions Once 🎯

Jobs are Runnable's solution for **one-time function execution**. Perfect for standalone tasks, testing, analysis, and automation scripts.

## What Are Jobs?

Jobs wrap your Python functions to provide:

- **Execution tracking** - Know what ran, when, and what happened
- **Parameter management** - Pass configuration from files or environment
- **Output capture** - Automatically store results and logs
- **Error handling** - Graceful failure management
- **Reproducibility** - Consistent execution across environments

## Quick Start

```python
from runnable import PythonJob

def analyze_data():
    """A simple analysis function."""
    data = [1, 2, 3, 4, 5]
    result = sum(data) / len(data)
    print(f"Average: {result}")
    return result

def main():
    job = PythonJob(function=analyze_data)
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

## Common Use Cases

### 📊 Data Analysis
Perfect for one-off analysis tasks:

```python
def monthly_sales_analysis():
    # run a notebook
```

### 🧪 Function Testing
Test your functions in isolation:

```python
def test_my_algorithm():
    # Test edge cases, validate outputs
    assert my_algorithm([1, 2, 3]) == expected_result
```

### 📝 Report Generation
Generate standalone reports:

```python
def generate_weekly_report():
    # Collect data, create visualizations, export PDF
    return "report_2024_week_47.pdf"
```

## Job Types

Runnable supports multiple job types for different scenarios:

| Job Type | Purpose | Example |
|----------|---------|---------|
| **PythonJob** | Execute Python functions | Data analysis, calculations |
| **NotebookJob** | Run Jupyter notebooks | Interactive analysis, reports |
| **ShellJob** | Execute shell commands | System operations, deployments |

## Key Features

### ✅ **Simple Execution**
```python
job = PythonJob(function=my_function)
job.execute()
```

### ⚙️ **Parameter Support**
```python
job.execute(parameters_file="config.yaml")
```

### 📁 **Automatic Output Storage**
- Execution logs captured automatically
- Results stored in catalog
- Reproducible execution history


## When to Use Jobs vs Pipelines

| Scenario | Use Jobs | Use Pipelines |
|----------|----------|---------------|
| Single function to run | ✅ | ❌ |
| Testing a function | ✅ | ❌ |
| One-off analysis | ✅ | ❌ |
| Multi-step workflow | ❌ | ✅ |
| Data dependencies between steps | ❌ | ✅ |
| Reproducible processes | ❌ | ✅ |

!!! tip "Start Simple, Grow Complex"

    Start with a Job to test your function, then evolve it into a Pipeline when you need multiple steps or complex workflows.

## What's Next?

- **[Getting Started](getting-started.md)** - Step-by-step guide to your first Job
- **[Configuration](configuration.md)** - Advanced Job options and settings
- **[Examples](examples.md)** - Real-world Job patterns and use cases
- **[Jobs vs Pipelines](../concepts/building-blocks/jobs-vs-pipelines.md)** - Detailed comparison and decision guide
