# 📊 Adding Your Data

Your functions probably create and return data. Runnable handles this automatically.

## Start with your return values

Here's a function that creates some data:

```python
def write_parameter():
    """A function that returns multiple parameters"""
    df = pd.DataFrame({"x": [1, 2, 3]})
    return df, 10, 3.14, "hello", SamplePydanticModel(x=10, foo="bar"), 0.95
```

## Job mode - Returns work automatically

In job mode, you can name your return values for better tracking and storage:

```python
from runnable import PythonJob, pickled, metric

job = PythonJob(
    function=write_parameter,
    returns=[
        pickled("df"),        # For pandas DataFrames
        "integer",            # Simple types work as-is
        "floater",
        "stringer",
        "pydantic_param",     # Pydantic models handled automatically
        metric("score")       # Mark metrics for monitoring
    ]
)
job.execute()
```

??? example "See complete runnable code"
    ```python title="examples/11-jobs/passing_parameters_python.py"
    --8<-- "examples/11-jobs/passing_parameters_python.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/11-jobs/passing_parameters_python.py
    ```

## Pipeline mode - Name your outputs

When your function is part of a workflow, name the outputs so other tasks can use them:

```python
from runnable import PythonTask, pickled, metric

task = PythonTask(
    function=write_parameter,
    returns=[
        pickled("df"),        # For pandas DataFrames
        "integer",            # Simple types work as-is
        "floater",
        "stringer",
        "pydantic_param",     # Pydantic models handled automatically
        metric("score")       # Mark metrics for monitoring
    ]
)
```

Now downstream tasks can access `df`, `integer`, `floater`, etc.

??? example "See complete runnable code"
    ```python title="examples/03-parameters/passing_parameters_python.py"
    --8<-- "examples/03-parameters/passing_parameters_python.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/03-parameters/passing_parameters_python.py
    ```

## Handle different data types

### 📦 Large or complex objects

Use `pickled()` for pandas DataFrames, models, or large collections:

```python
returns=[pickled("df"), "score"]
```

### 📈 Track metrics

Use `metric()` for numbers you want to monitor:

```python
returns=[metric("accuracy"), metric("loss")]
```

### 📋 Everything else

Simple types (strings, numbers, lists) work as-is:

```python
returns=["count", "status", "results"]
```

!!! tip "Pro tip"

    Name your returns clearly. `["df", "score"]` is better than `["output1", "output2"]`.

!!! info "Works with all task types"

    Everything you've learned here works identically with:

    - **Python functions** (`PythonTask`, `PythonJob`)
    - **Jupyter notebooks** (`NotebookTask`, `NotebookJob`)
    - **Shell scripts** (`ShellTask`, `ShellJob`)

    Same `returns=[]` syntax, same data types, same parameter passing. Only the execution environment changes!

Next: Learn how to [connect functions](connecting-functions.md) in workflows.
