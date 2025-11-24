# 🔄 Perfect Reproducibility Every Time

Tired of "it worked on my machine" problems? Runnable automatically captures everything needed to reproduce your workflows.

## The old way (hope and pray)

```python
def analyze_data():
    # Which version of pandas was this?
    # What were the input files?
    # Which git commit was this?
    df = pd.read_csv("data.csv")  # Hope it's the same data...
    return df.groupby('category').mean()  # Hope same pandas behavior...
```

## The Runnable way (automatic tracking)

**Every run captures:**
```python
from runnable import Pipeline, PythonTask

pipeline = Pipeline(steps=[
    PythonTask(function=analyze_data, name="analysis")
])
result = pipeline.execute()  # Everything is automatically tracked!
```

**After running, you get:**

- 🆔 **Unique run ID**: `clever-einstein-1234`
- 📝 **Complete execution log**: `.run_log_store/clever-einstein-1234.json`
- 💾 **All data artifacts**: `.catalog/clever-einstein-1234/`
- 🔍 **Full metadata**: Parameters, timings, code versions

## What gets tracked automatically

**Code & Environment:**
```json
{
  "code_identities": [{
    "code_identifier": "7079b8df5c4bf826d3baf6e3f5839ba6193d88dd",
    "code_identifier_type": "git",
    "code_identifier_url": "https://github.com/your-org/project.git"
  }]
}
```

**Parameters & Data Flow:**
```json
{
  "input_parameters": {"threshold": 0.95},
  "output_parameters": {"accuracy": 0.87},
  "data_catalog": [{
    "name": "model.pkl",
    "data_hash": "8650858600ce25b35e978ecb162414d9"
  }]
}
```

**Execution Context:**
```json
{
  "start_time": "2025-11-04 22:48:42.128195",
  "status": "SUCCESS",
  "pipeline_executor": {"service_name": "local"},
  "dag_hash": "d26e1287acb814e58c72a1c67914033eb0fb2e26"
}
```

## Complete workflow example

```python
from runnable import Pipeline, PythonTask, Catalog, pickled

def train_model(learning_rate: float = 0.01):
    model = train_ml_model(learning_rate)
    return {"model": model, "accuracy": 0.87}

def evaluate_model(model, test_data_path: str):
    accuracy = evaluate(model, test_data_path)
    return {"final_accuracy": accuracy}

pipeline = Pipeline(steps=[
    PythonTask(
        function=train_model,
        returns=[pickled("model"), ("accuracy", "json")],
        catalog=Catalog(get=["train.csv"], put=["model.pkl"])
    ),
    PythonTask(
        function=evaluate_model,
        catalog=Catalog(get=["test.csv"])
    )
])

# Everything gets tracked automatically
result = pipeline.execute()
print(f"Run ID: {result.run_id}")  # clever-einstein-1234
```

??? example "See complete runnable code"
    ```python title="examples/03-parameters/passing_parameters_python.py"
    --8<-- "examples/03-parameters/passing_parameters_python.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/03-parameters/passing_parameters_python.py
    ```

## Exploring your run history

**Find your run:**
```bash
ls .run_log_store/
# clever-einstein-1234.json
# nervous-tesla-5678.json
```

**Examine what happened:**
```python
import json
with open('.run_log_store/clever-einstein-1234.json') as f:
    run_log = json.load(f)

print(f"Status: {run_log['status']}")
print(f"Final accuracy: {run_log['parameters']['final_accuracy']}")
```

**Access the data:**
```bash
ls .catalog/clever-einstein-1234/
# model.pkl
# train.csv
# test.csv
# step1_execution.log
```

## Real example: Catalog tracking

Let's see how file management gets tracked:

```python
from runnable import Pipeline, PythonTask, Catalog

def generate_data():
    # Create files that will be tracked
    df.to_csv("df.csv")
    with open("data_folder/data.txt", "w") as f:
        f.write("Important data")

def process_data():
    # Files are automatically available here
    df = pd.read_csv("df.csv")
    with open("data_folder/data.txt") as f:
        content = f.read()

pipeline = Pipeline(steps=[
    PythonTask(
        function=generate_data,
        catalog=Catalog(put=["df.csv", "data_folder/data.txt"]),
        name="generate"
    ),
    PythonTask(
        function=process_data,
        catalog=Catalog(get=["df.csv", "data_folder/data.txt"]),
        name="process"
    )
])
pipeline.execute()
```

??? example "See complete runnable code"
    ```python title="examples/04-catalog/catalog_python.py"
    --8<-- "examples/04-catalog/catalog_python.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/04-catalog/catalog_python.py
    ```

## What gets logged for each step

**Step Summary:**
```json
{
  "Name": "generate_data",
  "Input catalog content": [],
  "Available parameters": [],
  "Output catalog content": ["df.csv", "data_folder/data.txt"],
  "Output parameters": [],
  "Metrics": [],
  "Code identities": ["git:7079b8df5c4bf826d3baf6e3f5839ba6193d88dd"],
  "status": "SUCCESS"
}
```

**File Tracking:**
```json
{
  "data_catalog": [
    {
      "name": "df.csv",
      "data_hash": "8650858600ce25b35e978ecb162414d9",
      "catalog_relative_path": "run-id-123/df.csv",
      "stage": "put"
    }
  ]
}
```

## Why this matters

**Without automatic tracking:**

- ❌ "It worked last week" debugging sessions
- ❌ Lost parameter combinations that worked
- ❌ No way to reproduce important results
- ❌ Manual documentation that gets stale

**With Runnable's tracking:**

- ✅ Every run is completely reproducible
- ✅ Compare results across different runs
- ✅ Debug with full execution context
- ✅ Zero-effort audit trails for compliance

## Advanced: Custom run tracking

```python
# Tag important runs
pipeline.execute(tag="production-candidate")

# Environment-specific tracking
pipeline.execute(config="configs/production.yaml")
```

### Custom Run IDs via Environment

Control pipeline execution tracking with custom identifiers:

```bash
# Set custom run ID for tracking and debugging
export RUNNABLE_RUN_ID="experiment-learning-rate-001"
uv run ml_pipeline.py

# Daily ETL runs with dates
export RUNNABLE_RUN_ID="etl-daily-$(date +%Y-%m-%d)"
uv run data_pipeline.py

# Experiment tracking with git context
export RUNNABLE_RUN_ID="experiment-$(git branch --show-current)-v2"
uv run research_pipeline.py
```

**Benefits for reproducibility:**

- **Predictable naming** for experiment tracking
- **Easy identification** in run history and logs
- **Integration** with external systems and CI/CD
- **Consistent tracking** across related pipeline executions

## Run ID patterns

Runnable generates memorable run IDs automatically:

- `obnoxious-williams-2248` - From our catalog example
- `nervous-sinoussi-2248` - From our parameters example
- `clever-einstein-1234` - Hypothetical example

Each ID is unique and helps you easily reference specific runs in conversations and debugging.

!!! tip "Best practices"

    - Let Runnable generate run IDs for exploration
    - Use tags for important experimental runs
    - Keep your git repo clean for reliable code tracking
    - Use the catalog for all data that flows between steps

Next: Learn how to [deploy anywhere](../production/deploy-anywhere.md) while keeping the same reproducibility guarantees.
