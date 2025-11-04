# 🎯 Jobs vs Pipelines: When to Use Which?

Both jobs and pipelines run your functions. The difference is **intent**.

## 🎯 Jobs: "Run this once"

Perfect for standalone tasks:

```python linenums="1"
def analyze_sales_data():
    # Load data, run analysis, generate report
    pass

# Job: Just run it
job = PythonJob(function=analyze_sales_data)
result = job.execute()
```

### When to use jobs:

- **One-off analysis**: "Analyze this dataset"
- **Testing functions**: "Does my code work?"
- **Standalone reports**: "Generate monthly summary"
- **Data exploration**: "What's in this file?"

## 🔗 Pipelines: "This is step X of many"

Perfect for multi-step workflows:

```python linenums="1"
def load_data():
    return raw_data

def clean_data(raw_data):
    return cleaned_data

def train_model(cleaned_data):
    return model

# Pipeline: Chain them together
pipeline = Pipeline(steps=[
    PythonTask(function=load_data, returns=["raw_data"]),
    PythonTask(function=clean_data, returns=["cleaned_data"]),
    PythonTask(function=train_model, returns=["model"])
])
```

### When to use pipelines:

- **Multi-step workflows**: "Load → Clean → Train → Deploy"
- **Data pipelines**: "Extract → Transform → Load"
- **Reproducible processes**: "Run the same steps every time"
- **Complex dependencies**: "Step 3 needs outputs from steps 1 and 2"

## 🔄 Same function, different contexts

Here's the same function used both ways:

```python linenums="1"
--8<-- "examples/common/functions.py:14:18"
```

**As a job** (from `examples/11-jobs/python_tasks.py`):
```python linenums="1"
--8<-- "examples/11-jobs/python_tasks.py:7:13"
```

**As a pipeline task** (from `examples/01-tasks/python_tasks.py`):
```python linenums="1"
--8<-- "examples/01-tasks/python_tasks.py:7:17"
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

Next: Learn about [different task types](task-types.md) (Python, notebooks, shell scripts).
