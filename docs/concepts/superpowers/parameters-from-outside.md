# ⚡ Parameters From Outside

The coolest part: Change your function's behavior without changing your code.

## Your function accepts parameters

```python linenums="1"
--8<-- "examples/common/functions.py:45:58"
```

## Method 1: YAML files

Create a parameters file:

```yaml title="parameters.yaml"
--8<-- "examples/common/initial_parameters.yaml"
```

Run with parameters:

```python linenums="1"
--8<-- "examples/03-parameters/static_parameters_python.py:56:62"
```

## Method 2: Environment variables

Set variables with `RUNNABLE_PRM_` prefix:

```bash
export RUNNABLE_PRM_integer=42
export RUNNABLE_PRM_stringer="production data"
```

```python linenums="1"
--8<-- "examples/03-parameters/static_parameters_python.py:65:69"
```

## 🏆 Environment variables win

If you have both YAML and environment variables, environment variables take priority:

```python linenums="1"
--8<-- "examples/03-parameters/static_parameters_python.py:65:69"
```

YAML says `envvar: "not set"`, but environment variable `RUNNABLE_PRM_envvar="from env"` wins.

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
