# ⚡ Parameters From Outside

The coolest part: Change your function's behavior without changing your code.

## Your function accepts parameters

```python
def read_initial_params_as_pydantic(
    integer: int,
    floater: float,
    stringer: str,
    pydantic_param: ComplexParams,
    envvar: str,
):
    # Your function uses these parameters
    print(f"Processing {integer} items with {stringer} config")
    return f"Processed {len(pydantic_param)} records"
```

## Method 1: YAML files

Create a parameters file:

```yaml title="parameters.yaml"
integer: 1
floater: 3.14
stringer: "hello"
pydantic_param:
  x: 10
  foo: "bar"
envvar: "not set"  # Will be overridden by environment variable
```

Run with parameters:

```python
from runnable import Pipeline, PythonTask

task = PythonTask(function=read_initial_params_as_pydantic)
pipeline = Pipeline(steps=[task])
pipeline.execute(parameters_file="parameters.yaml")
```

??? example "See complete runnable code"
    ```python title="examples/03-parameters/static_parameters_python.py"
    --8<-- "examples/03-parameters/static_parameters_python.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/03-parameters/static_parameters_python.py
    ```

## Method 2: Environment variables

Set variables with `RUNNABLE_PRM_` prefix:

```bash
export RUNNABLE_PRM_integer=42
export RUNNABLE_PRM_stringer="production data"
export RUNNABLE_PRM_envvar="from env"  # This overrides YAML!
```

```python
import os

# Set environment variable in code (for demo)
os.environ["RUNNABLE_PRM_envvar"] = "from env"

# Same pipeline execution as before
pipeline.execute(parameters_file="parameters.yaml")
```

## 🏆 Environment variables win

If you have both YAML and environment variables, environment variables take priority:

- **YAML file says:** `envvar: "not set"`
- **Environment variable:** `RUNNABLE_PRM_envvar="from env"`
- **Result:** Your function gets `"from env"` ✅

## Why this is powerful

**Same code, different behavior:**

```python
# Development
export RUNNABLE_PRM_dataset="small_test_data.csv"
pipeline.execute()

# Production
export RUNNABLE_PRM_dataset="full_production_data.csv"
pipeline.execute()
```

**No code changes needed!**

## Complex parameters work too

```python
# Nested objects
export RUNNABLE_PRM_model_config='{"learning_rate": 0.01, "epochs": 100}'

# Lists
export RUNNABLE_PRM_features='["age", "income", "location"]'
```

!!! tip "Pro tip"

    Use YAML files for default parameters, environment variables for overrides. Perfect for different environments (dev/staging/prod).

Next: Learn about [automatic file management](file-management.md) between tasks.
