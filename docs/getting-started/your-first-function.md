# 🚀 From Python Function to Runnable

Got a Python function? Great! You're 90% of the way to using Runnable.

## Start with what you know

Here's a regular Python function:

```python
def hello():
    "The most basic function"
    print("Hello World!")
```

You probably run it like this:

```python
if __name__ == "__main__":
    hello()
```

## Two ways to run it with Runnable

### 🎯 Job mode - Single execution

Perfect for one-off tasks. The key concept is wrapping your function:

```python
from runnable import PythonJob

job = PythonJob(function=hello)
job.execute()
```

That's it! Your function runs exactly the same, but now Runnable handles execution and logging.

??? example "See complete runnable code"
    ```python title="examples/11-jobs/python_tasks.py"
    --8<-- "examples/11-jobs/python_tasks.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/11-jobs/python_tasks.py
    ```

### 🔗 Pipeline mode - Part of a workflow

Perfect when this function is one step in a larger process:

```python
from runnable import Pipeline, PythonTask

task = PythonTask(function=hello, name="say_hello")
pipeline = Pipeline(steps=[task])
pipeline.execute()
```

??? example "See complete runnable code"
    ```python title="examples/01-tasks/python_tasks.py"
    --8<-- "examples/01-tasks/python_tasks.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/01-tasks/python_tasks.py
    ```

## That's it!

Same function, same output. Runnable just gives you two execution modes:

- **Job**: "Run this function once"
- **Pipeline**: "This function is step 1 of many"

!!! tip "When to use which?"

    - **Job** → Standalone analysis, one-off data processing, testing functions
    - **Pipeline** → Multi-step workflows, data pipelines, ML training sequences

Next: Learn how to [handle your data](handling-data.md) with return values.
