# Working with Data 📦

Learn how to store and return data from your Jobs.

## Returning Data from Functions

When your function returns data, specify what should be stored using the `returns` parameter:

```python
from examples.common.functions import write_parameter
from runnable import PythonJob, metric, pickled

def main():
    job = PythonJob(
        function=write_parameter,
        returns=[
            pickled("df"),        # pandas DataFrame (complex object)
            "integer",            # JSON-serializable integer
            "floater",            # JSON-serializable float
            "stringer",           # JSON-serializable string
            "pydantic_param",     # Pydantic model (auto-handled)
            metric("score"),      # Metric for tracking
        ],
    )

    job.execute()
    return job

if __name__ == "__main__":
    main()
```

??? example "See complete runnable code"
    ```python title="examples/11-jobs/passing_parameters_python.py"
    --8<-- "examples/11-jobs/passing_parameters_python.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/11-jobs/passing_parameters_python.py
    ```

## What Gets Stored

```json
{
    "Output parameters": [
        ("df", "Pickled object stored in catalog as: df"),
        ("integer", 1),
        ("floater", 3.14),
        ("stringer", "hello"),
        ("pydantic_param", {"x": 10, "foo": "bar"}),
        ("score", 0.9)
    ],
    "Metrics": [("score", 0.9)],
    "status": "SUCCESS"
}
```

## Return Type Guide

| Type | Usage | Storage Location | Example |
|------|-------|-----------------|---------|
| `pickled("name")` | Complex objects (DataFrames, models) | `.catalog/{run-id}/name.dill` | `pickled("model")` |
| `"name"` | JSON-serializable (int, float, str, dict) | Job summary | `"count"` |
| `metric("name")` | Trackable metrics | Metrics section + summary | `metric("accuracy")` |
| Pydantic models | Auto-handled objects | Job summary as JSON | `"user_profile"` |

## Practical Examples

### Data Analysis Job
```python
def analyze_sales():
    # Your analysis logic here
    summary = {"total_sales": 50000, "growth": 0.15}
    return summary

job = PythonJob(
    function=analyze_sales,
    returns=["summary"]
)
```

### Model Training Job
```python
def train_model():
    # Training logic here
    model = create_model()
    accuracy = 0.95
    return model, accuracy

job = PythonJob(
    function=train_model,
    returns=[pickled("model"), metric("accuracy")]
)
```

### Report Generation Job
```python
def generate_report():
    # Report logic here
    report_path = "monthly_report.pdf"
    metrics = {"pages": 12, "charts": 5}
    return report_path, metrics

job = PythonJob(
    function=generate_report,
    returns=["report_path", "metrics"]
)
```

## Viewing Stored Data

After execution, check what was stored:

```bash
# List catalog contents
ls .catalog/{run-id}/

# View pickled objects (requires Python)
# Complex objects are in .dill files

# Simple values appear in job summary
# Check terminal output for JSON summary
```

## Best Practices

### ✅ **Always Specify Returns**
```python
# Good - explicit about what to keep
job = PythonJob(
    function=my_function,
    returns=["result", metric("score")]
)
```

### ❌ **Don't Forget Returns**
```python
# Bad - function output will be lost
job = PythonJob(function=my_function)  # No returns specified!
```

### ✅ **Use Appropriate Types**
```python
returns=[
    pickled("large_dataframe"),  # For complex objects
    "simple_count",              # For basic values
    metric("accuracy"),          # For trackable metrics
]
```

## What's Next?

You can now store Job outputs! Next steps:

- **[Parameters & Environment](parameters.md)** - Configure Jobs dynamically
- **[File Storage](file-storage.md)** - Store files created during execution
- **[Job Types](job-types.md)** - Shell and Notebook Jobs

Ready to make your Jobs configurable? Continue to **[Parameters & Environment](parameters.md)**!
