# Job Types

Explore different types of Jobs for various execution scenarios.

## Shell Jobs 🔧

Execute shell commands with full context tracking:

```python
from runnable import ShellJob

def main():
    job = ShellJob(command="echo 'Hello World!'")
    job.execute()
    return job

if __name__ == "__main__":
    main()
```

??? example "See complete runnable code"
    ```python title="examples/11-jobs/scripts.py"
    --8<-- "examples/11-jobs/scripts.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/11-jobs/scripts.py
    ```

### Shell Job Features

- **Command output**: Stdout captured in execution logs
- **Environment variables**: `RUNNABLE_RUN_ID` and `PWD` automatically available
- **Exit codes**: Success/failure tracked automatically

### Common Shell Job Patterns

```python
# File processing
ShellJob(command="python data_processing.py input.csv output.csv")

# System operations
ShellJob(command="tar -czf backup.tar.gz /data")

# External tools
ShellJob(command="curl -X POST https://api.example.com/webhook")

# Data pipeline steps
ShellJob(command="./preprocess_data.sh && ./run_analysis.sh")

# Docker containers
ShellJob(command="docker run --rm -v $(pwd):/data my-analysis:latest")
```

### Shell Job with Parameters

Shell Jobs support the same parameter system:

```python
def main():
    job = ShellJob(command="python analysis.py --input {input_file} --output {output_dir}")
    job.execute(parameters_file="config.yaml")
    return job
```

**config.yaml:**
```yaml
input_file: "data/sales.csv"
output_dir: "results/"
```

## Notebook Jobs 📓

Execute Jupyter notebooks with complete output preservation:

```python
from runnable import NotebookJob

def main():
    job = NotebookJob(notebook="examples/common/simple_notebook.ipynb")
    job.execute()
    return job

if __name__ == "__main__":
    main()
```

??? example "See complete runnable code"
    ```python title="examples/11-jobs/notebooks.py"
    --8<-- "examples/11-jobs/notebooks.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/11-jobs/notebooks.py
    ```

### Notebook Job Features

- **Executed notebook**: Complete notebook with outputs saved as `{notebook}-_out.ipynb`
- **Directory structure**: Maintains original path in catalog
- **Cell outputs**: All print statements, plots, and results captured
- **Progress tracking**: Shows cell execution progress during run

### Notebook Storage Structure

```
.catalog/dry-lichterman-0628/
├── examples/
│   └── common/
│       └── simple_notebook-_out.ipynb  # Executed notebook with outputs
└── jobXLV.execution.log                # Execution log
```

### Notebook Job Patterns

```python
# Data analysis notebook
NotebookJob(notebook="analysis/monthly_report.ipynb")

# Model training notebook
NotebookJob(notebook="ml/train_model.ipynb")

# Visualization notebook
NotebookJob(notebook="reports/create_charts.ipynb")

# Research notebook with parameters
job = NotebookJob(notebook="research/experiment.ipynb")
job.execute(parameters_file="experiment_config.yaml")
```

### Notebooks with Parameters

Notebooks can receive parameters through parameter injection:

**notebook.ipynb** (cell 1):
```python
# Parameters will be injected here automatically
dataset_path = "default.csv"  # Will be overridden
model_type = "random_forest"  # Will be overridden
```

**Job execution:**
```python
def main():
    job = NotebookJob(notebook="analysis.ipynb")
    job.execute(parameters={
        "dataset_path": "prod_data.csv",
        "model_type": "gradient_boost"
    })
    return job
```

## Python Jobs (Review) 🐍

The most flexible Job type for custom functions:

```python
from runnable import PythonJob

def my_analysis():
    # Custom analysis logic
    return {"result": "success"}

def main():
    job = PythonJob(
        function=my_analysis,
        returns=["result"]
    )
    job.execute()
    return job
```

### Python Job Advantages

- **Full flexibility**: Complete control over execution logic
- **Return values**: Specify exactly what to store
- **Type safety**: Function parameters validated automatically
- **Debugging**: Easy to test functions independently

## Choosing the Right Job Type

| Use Case | Best Job Type | Why |
|----------|---------------|-----|
| Custom data analysis | **PythonJob** | Full control, return values, type safety |
| System operations | **ShellJob** | Leverage existing scripts and tools |
| Interactive analysis | **NotebookJob** | Visual outputs, step-by-step exploration |
| File processing | **ShellJob** | Use command-line tools directly |
| Model training | **PythonJob** or **NotebookJob** | Python for code, Notebook for exploration |
| Report generation | **NotebookJob** | Rich outputs with plots and formatting |
| Data pipeline steps | **ShellJob** | Chain existing pipeline tools |
| API integration | **PythonJob** | Handle responses and error cases |

## Combining Job Types

Different Jobs can work together in larger workflows:

```python
# Step 1: Preprocess with shell tools
preprocessing = ShellJob(command="./preprocess_data.sh")

# Step 2: Analysis with Python
def analyze_data():
    # Load preprocessed data and analyze
    return results

analysis = PythonJob(function=analyze_data, returns=["results"])

# Step 3: Report with notebook
reporting = NotebookJob(notebook="generate_report.ipynb")
```

## Job Execution Context 🔍

All Job types share the same rich execution context:

```
JobContext(
    execution_mode='python',           # or 'shell', 'notebook'
    run_id='feasible-booth-0628',      # Unique execution ID
    catalog=FileSystemCatalog(         # File storage
        catalog_location='.catalog'
    ),
    job_executor=LocalJobExecutor(),   # Execution environment
    secrets=EnvSecretsManager(),       # Secret management
    run_log_store=FileSystemRunLogstore()  # Metadata storage
)
```

## Best Practices

### ✅ **Choose Based on Task Nature**
```python
# Data science exploration -> Notebook
NotebookJob(notebook="explore_data.ipynb")

# Production analysis function -> Python
PythonJob(function=production_analysis)

# System integration -> Shell
ShellJob(command="./deploy_model.sh")
```

### ✅ **Use Parameters for Flexibility**
```python
# All Job types support parameters
job.execute(parameters_file="config.yaml")

# Environment variables work across all types
# export RUNNABLE_PRM_environment=production
```

### ✅ **Organize Outputs Appropriately**
```python
# Python Job - specify returns
PythonJob(function=analysis, returns=["results", metric("accuracy")])

# Shell/Notebook Jobs - use catalog for files
catalog = Catalog(put=["output.csv", "plots/"])
ShellJob(command="./analysis.sh", catalog=catalog)
```

## What's Next?

You now understand all Job types! Final topics:

- **[Configuration](configuration.md)** - Advanced Job options and settings
- **[Examples](examples.md)** - Real-world Job patterns and use cases
- **[Jobs vs Pipelines](../concepts/building-blocks/jobs-vs-pipelines.md)** - When to move to multi-step workflows

Ready for advanced configuration? Continue to **[Configuration](configuration.md)**!
