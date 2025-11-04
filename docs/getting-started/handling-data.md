# 📊 Adding Your Data

Your functions probably create and return data. Runnable handles this automatically.

## Start with your return values

Here's a function that creates some data:

```python linenums="1"
--8<-- "examples/common/functions.py:73:82"
```

## Job mode - Returns work automatically

```python linenums="1"
--8<-- "examples/11-jobs/passing_parameters_python.py"
```

Your return values are automatically captured and stored.

## Pipeline mode - Name your outputs

When your function is part of a workflow, name the outputs so other tasks can use them:

```python linenums="1"
--8<-- "examples/03-parameters/passing_parameters_python.py:26:37"
```

Now downstream tasks can access `df`, `integer`, `floater`, etc.

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

Next: Learn how to [connect functions](connecting-functions.md) in workflows.
