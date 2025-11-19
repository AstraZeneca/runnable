# 📁 File Management Made Easy

Tired of managing temporary files between tasks? Runnable's catalog system handles it automatically **and** gives you complete execution traceability.

## The old way (manual file management)

```python
def create_report():
    df = analyze_data()
    df.to_csv("temp_results.csv")  # Hope this exists later...

def send_report():
    df = pd.read_csv("temp_results.csv")  # Hope this file is there...
    # What if the path changed? What if step 1 failed?
```

## The Runnable way (automatic)

**Step 1: Create and store files**

```python
from runnable import Catalog, PythonTask

def write_files():
    # Create your files
    df.to_csv("df.csv")
    with open("data_folder/data.txt", "w") as f:
        f.write("Important data")

# Store files automatically
task1 = PythonTask(
    function=write_files,
    catalog=Catalog(put=["df.csv", "data_folder/data.txt"])
)
```

**Step 2: Retrieve and use files**

```python
def read_files():
    # Files are automatically available here!
    df = pd.read_csv("df.csv")  # ✅ File is there
    with open("data_folder/data.txt") as f:
        data = f.read()  # ✅ File is there

# Get files automatically
task2 = PythonTask(
    function=read_files,
    catalog=Catalog(get=["df.csv", "data_folder/data.txt"])
)
```

## How it works

1. **`put=["file.csv"]`** → Runnable stores the file safely
2. **`get=["file.csv"]`** → Runnable makes the file available in the next task
3. **No path management** → Files appear where your code expects them

## Full workflow example

```python
from runnable import Pipeline, PythonTask, Catalog

def main():
    # Complete workflow with automatic file management
    pipeline = Pipeline(steps=[
        PythonTask(
            function=write_files,
            catalog=Catalog(put=["df.csv", "data_folder/data.txt"]),
            name="create_files"
        ),
        PythonTask(
            function=read_files,
            catalog=Catalog(get=["df.csv", "data_folder/data.txt"]),
            name="process_files"
        )
    ])
    pipeline.execute()
    return pipeline

if __name__ == "__main__":
    main()
```

??? example "See complete runnable code"
    ```python title="examples/04-catalog/catalog.py"
    --8<-- "examples/04-catalog/catalog.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/04-catalog/catalog.py
    ```

## Multiple files and folders

```python
# Store multiple files explicitly
catalog=Catalog(put=["results.csv", "plots/", "model.pkl"])

# Retrieve what you need
catalog=Catalog(get=["results.csv", "model.pkl"])
```

## No-Copy Mode for Large Files 🚀

For large files or datasets, copying can be expensive and unnecessary. Use `store_copy=False` to track files without copying them:

```python
# Large dataset processing - track but don't copy
task1 = PythonTask(
    function=process_large_dataset,
    catalog=Catalog(put=["large_dataset.parquet", "model.pkl"], store_copy=False)
)

# Next task can still access the files
task2 = PythonTask(
    function=analyze_results,
    catalog=Catalog(get=["large_dataset.parquet"])
)
```

**What happens with `store_copy=False`:**

- ✅ **MD5 hash captured** for integrity verification
- ✅ Files remain in original location
- ✅ **No disk space duplication** for large files
- ✅ **Faster execution** - no time spent copying
- ✅ Still tracked in pipeline execution history

**When to use no-copy mode:**

- **Large datasets** (GB+ files) where copying is slow and expensive
- **Reference data** that doesn't change and is already stored safely
- **Network storage** where files are already backed up
- **Performance-critical pipelines** where copy time matters

**Example with mixed copy strategies:**

```python
pipeline = Pipeline(steps=[
    PythonTask(
        function=prepare_data,
        catalog=Catalog(put=[
            "config.json",           # Small file - copy it
            "large_input.parquet"    # Large file - hash only
        ], store_copy=False),        # Applies to all files
        name="prepare"
    ),
    PythonTask(
        function=process_data,
        catalog=Catalog(get=["config.json", "large_input.parquet"]),
        name="process"
    )
])
```

## Glob-style wildcards

Use wildcards to match multiple files automatically:

```python
# Store all CSV files
catalog=Catalog(put=["*.csv"])

# Store all files in data folder
catalog=Catalog(put=["data/*"])

# Store all Python files recursively
catalog=Catalog(put=["**/*.py"])

# Store all files with specific pattern
catalog=Catalog(put=["results_*.json", "plots/*.png"])
```

**Common wildcard patterns:**

| Pattern | Matches |
|---------|---------|
| `*.csv` | All CSV files in current directory |
| `data/*` | All files in the data folder |
| `**/*.py` | All Python files in current and subdirectories |
| `results_*.json` | Files like `results_train.json`, `results_test.json` |
| `plots/*.png` | All PNG files in the plots folder |

**Example with wildcards:**

```python
def main():
    pipeline = Pipeline(steps=[
        PythonTask(
            function=create_multiple_outputs,
            catalog=Catalog(put=["*.csv", "plots/*.png"]),  # Store all CSVs and plot PNGs
            name="generate_data"
        ),
        PythonTask(
            function=process_outputs,
            catalog=Catalog(get=["data_*.csv", "plots/summary.png"]),  # Get specific files
            name="process_data"
        )
    ])
    pipeline.execute()
    return pipeline

if __name__ == "__main__":
    main()
```

## Why this matters

**Without catalog:**

- ❌ Manual path management
- ❌ Files get lost between environments
- ❌ Hard to reproduce workflows
- ❌ Cleanup is manual

**With catalog:**

- ✅ Automatic file management
- ✅ Works across different environments
- ✅ Perfect reproducibility
- ✅ Automatic cleanup

## Automatic execution traceability

Runnable automatically captures all execution outputs in the `catalog` for complete traceability:

### What gets captured

**For every task execution, Runnable stores:**

1. **Execution logs** - Complete stdout/stderr output from your tasks
2. **Output notebooks** - Executed notebooks with all outputs and results (for NotebookTask)
3. **Environment information** - Environment variables and execution context
4. **Timestamps** - Precise execution timing information

### Where to find execution outputs

```bash
.catalog/
└── {run-id}/                          # Unique run identifier
    ├── taskname123.execution.log       # Task stdout/stderr output
    ├── output_notebook.ipynb          # Executed notebook (if NotebookTask)
    └── data_files/                    # Your catalog files
```

**Example after running a pipeline:**

```bash
.catalog/
└── pleasant-nobel-2303/
    ├── hello_task.execution.log        # "Hello World!" output captured
    ├── data_processing.execution.log   # All Python print statements
    ├── analysis_notebook_out.ipynb     # Executed notebook with results
    └── results.csv                     # Your data files
```

### Complete execution visibility

**Python tasks:** Capture all `print()`, `logging`, warnings, and errors:
```bash
$ cat .catalog/run-id/python_task.execution.log
[23:02:46] Parameters available for the execution:
           {'input_file': 'data.csv'}
Processing 1000 rows...
Model accuracy: 94.2%
WARNING: Low confidence predictions detected
```

**Notebook tasks:** Store executed notebooks with all outputs:

- Input notebook: `analysis.ipynb`
- Output notebook: `.catalog/run-id/analysis_out.ipynb` (with all cell outputs)
- Execution log: `.catalog/run-id/notebook_task.execution.log`

**Shell tasks:** Capture all command output and environment:
```bash
$ cat .catalog/run-id/shell_task.execution.log
Installing dependencies...
Running analysis script...
=== COLLECT ===
RUNNABLE_RUN_ID=run-id
PWD=/path/to/project
Results saved to output.txt
```

### Why this matters for debugging

**No more digging through logs!** Everything is organized by run ID:

```python
def main():
    pipeline = Pipeline(steps=[
        PythonTask(function=data_processing, name="process"),
        NotebookTask(notebook="analysis.ipynb", name="analyze"),
        ShellTask(command="./deploy.sh", name="deploy")
    ])
    result = pipeline.execute()

    # Check .catalog/{run_id}/ for complete execution trace:
    # - process123.execution.log (Python output)
    # - analyze456_out.ipynb (executed notebook)
    # - deploy789.execution.log (shell output)

    return pipeline

if __name__ == "__main__":
    main()
```

!!! tip "Best practices"

    - Use catalog for files that flow between tasks. Keep truly temporary files local.
    - Use wildcards (`*.csv`, `data/*`) to automatically capture multiple files without manual listing.
    - Be specific with wildcards to avoid capturing unwanted files (`results_*.csv` vs `*.csv`).
    - **Use `store_copy=False` for large files** to save disk space and improve performance.
    - **Check `.catalog/{run-id}/` for complete execution traceability** - no need to dig through environment-specific logs!

Next: See how the same code can [run anywhere](deploy-anywhere.md) with different configurations.
