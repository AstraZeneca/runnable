# 🎯 Jobs vs Pipelines: When to Use Which?

Both jobs and pipelines run your functions. The difference is **intent**.

## 🎯 Jobs: "Run this once"

Perfect for standalone tasks:

```python
from runnable import PythonJob

def analyze_sales_data():
    # Load data, run analysis, generate report
    return "Analysis complete!"

def main():
    # Job: Just run it
    job = PythonJob(function=analyze_sales_data)
    result = job.execute()
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

### When to use jobs:

- **One-off analysis**: "Analyze this dataset"
- **Testing functions**: "Does my code work?"
- **Standalone reports**: "Generate monthly summary"
- **Data exploration**: "What's in this file?"

## 🔗 Pipelines: "This is step X of many"

Perfect for multi-step workflows:

```python
from runnable import Pipeline, PythonTask

def load_data():
    return {"users": 1000, "sales": 50000}

def clean_data(raw_data):
    return {"clean_users": raw_data["users"], "clean_sales": raw_data["sales"]}

def train_model(cleaned_data):
    return f"Model trained on {cleaned_data['clean_users']} users"

def main():
    # Pipeline: Chain them together
    pipeline = Pipeline(steps=[
        PythonTask(function=load_data, returns=["raw_data"]),
        PythonTask(function=clean_data, returns=["cleaned_data"]),
        PythonTask(function=train_model, returns=["model"])
    ])
    pipeline.execute()
    return pipeline

if __name__ == "__main__":
    main()
```

??? example "See complete runnable code"
    ```python title="examples/03-parameters/passing_parameters_python.py"
    --8<-- "examples/03-parameters/passing_parameters_python.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/03-parameters/passing_parameters_python.py
    ```

### When to use pipelines:

- **Multi-step workflows**: "Load → Clean → Train → Deploy"
- **Data pipelines**: "Extract → Transform → Load"
- **Reproducible processes**: "Run the same steps every time"
- **Complex dependencies**: "Step 3 needs outputs from steps 1 and 2"

## 🔄 Same function, different contexts

Here's the same function used both ways:

```python
def hello():
    "The most basic function"
    print("Hello World!")
```

**As a job:**
```python
from runnable import PythonJob

def main():
    job = PythonJob(function=hello)
    job.execute()
    return job

if __name__ == "__main__":
    main()
```

**As a pipeline task:**
```python
from runnable import Pipeline, PythonTask

def main():
    task = PythonTask(function=hello, name="say_hello")
    pipeline = Pipeline(steps=[task])
    pipeline.execute()
    return pipeline

if __name__ == "__main__":
    main()
```

## Quick decision guide

| I want to... | Use |
|--------------|-----|
| Test my function | Job |
| Run analysis once | Job |
| Generate a report | Job |
| Process data in multiple steps | Pipeline |
| Chain different functions | Pipeline |
| Run the same workflow repeatedly | Pipeline |

!!! tip "You can always switch"

    Start with a job to test your function, then move it into a pipeline when you're ready to build a workflow.

## What's Next?

- **[Pipeline Parameters](pipeline-parameters.md)** - Configure pipelines with parameters and custom run IDs
- **[Task Types](task-types.md)** - Different ways to define pipeline steps (Python, notebooks, shell scripts)
