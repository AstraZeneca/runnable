# 📁 File Management Made Easy

Tired of managing temporary files between tasks? Runnable's catalog system handles it automatically.

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
# Store multiple files
catalog=Catalog(put=["results.csv", "plots/", "model.pkl"])

# Retrieve what you need
catalog=Catalog(get=["results.csv", "model.pkl"])
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

!!! tip "Best practice"

    Use catalog for files that flow between tasks. Keep truly temporary files local.

Next: See how the same code can [run anywhere](deploy-anywhere.md) with different configurations.
