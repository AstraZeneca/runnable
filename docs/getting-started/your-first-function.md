# 🚀 From Python Function to Runnable

Got a Python function? Great! You're 90% of the way to using Runnable.

## Start with what you know

Here's a regular Python function:

```python linenums="1"
--8<-- "examples/common/functions.py:14:16"
```

You probably run it like this:

```python
if __name__ == "__main__":
    hello()
```

## Two ways to run it with Runnable

### 🎯 Job mode - Single execution

Perfect for one-off tasks:

```python linenums="1"
--8<-- "examples/11-jobs/python_tasks.py:25:28"
```

### 🔗 Pipeline mode - Part of a workflow

Perfect when this function is one step in a larger process:

```python linenums="1"
--8<-- "examples/01-tasks/python_tasks.py:18:26"
```

## That's it!

Same function, same output. Runnable just gives you two execution modes:

- **Job**: "Run this function once"
- **Pipeline**: "This function is step 1 of many"

!!! tip "When to use which?"

    - **Job** → Standalone analysis, one-off data processing, testing functions
    - **Pipeline** → Multi-step workflows, data pipelines, ML training sequences

Next: Learn how to [handle your data](handling-data.md) with return values.
