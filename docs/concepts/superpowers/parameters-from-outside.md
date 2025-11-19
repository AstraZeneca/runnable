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

def main():
    task = PythonTask(function=read_initial_params_as_pydantic)
    pipeline = Pipeline(steps=[task])
    pipeline.execute(parameters_file="parameters.yaml")
    return pipeline

if __name__ == "__main__":
    main()
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
from runnable import Pipeline, PythonTask

def main():
    task = PythonTask(function=read_initial_params_as_pydantic)
    pipeline = Pipeline(steps=[task])

    # Same pipeline execution as before
    pipeline.execute(parameters_file="parameters.yaml")
    return pipeline

if __name__ == "__main__":
    main()
```

## 🏆 Environment variables win

If you have both YAML and environment variables, environment variables take priority:

- **YAML file says:** `envvar: "not set"`
- **Environment variable:** `RUNNABLE_PRM_envvar="from env"`
- **Result:** Your function gets `"from env"` ✅

## Why this is powerful

**Same code, different behavior:**

```bash
# Option 1: Individual parameter overrides
export RUNNABLE_PRM_dataset="small_test_data.csv"
uv run my_pipeline.py

export RUNNABLE_PRM_dataset="full_production_data.csv"
uv run my_pipeline.py

# Option 2: Complete parameter file switching
export RUNNABLE_PARAMETERS_FILE="configs/dev.yaml"
uv run my_pipeline.py

export RUNNABLE_PARAMETERS_FILE="configs/prod.yaml"
uv run my_pipeline.py
```

**No code changes needed!**

## Method 3: Dynamic parameter files

Switch parameter files without changing code using `RUNNABLE_PARAMETERS_FILE`:

```bash
# Use different parameter files for different environments
export RUNNABLE_PARAMETERS_FILE="dev-parameters.yaml"
export RUNNABLE_PARAMETERS_FILE="prod-parameters.yaml"
export RUNNABLE_PARAMETERS_FILE="staging-parameters.yaml"
```

```python
from runnable import Pipeline, PythonTask

def main():
    task = PythonTask(function=read_initial_params_as_pydantic)
    pipeline = Pipeline(steps=[task])

    # No need to specify parameters_file - uses RUNNABLE_PARAMETERS_FILE
    pipeline.execute()
    return pipeline

if __name__ == "__main__":
    main()
```

**Powerful deployment pattern:**
```bash
# Development
export RUNNABLE_PARAMETERS_FILE="configs/dev.yaml"
uv run my_pipeline.py

# Production
export RUNNABLE_PARAMETERS_FILE="configs/prod.yaml"
uv run my_pipeline.py  # Same code, different parameters!
```

## Complex parameters work too

```python
# Nested objects
export RUNNABLE_PRM_model_config='{"learning_rate": 0.01, "epochs": 100}'

# Lists
export RUNNABLE_PRM_features='["age", "income", "location"]'
```

!!! tip "Pro tip"

    **Three-layer flexibility:**

    1. **Code-specified**: `pipeline.execute(parameters_file="base.yaml")`
    2. **Environment override**: `RUNNABLE_PARAMETERS_FILE="prod.yaml"` (overrides code)
    3. **Individual parameters**: `RUNNABLE_PRM_key="value"` (overrides both)

    Perfect for different environments (dev/staging/prod) without code changes!

Next: Learn about [automatic file management](file-management.md) between tasks.
