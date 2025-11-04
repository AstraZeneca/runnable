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

```python linenums="1"
--8<-- "examples/04-catalog/catalog.py:29:35"
```

**Step 2: Retrieve and use files**

```python linenums="1"
--8<-- "examples/04-catalog/catalog.py:37:43"
```

## How it works

1. **`put=["file.csv"]`** → Runnable stores the file safely
2. **`get=["file.csv"]`** → Runnable makes the file available in the next task
3. **No path management** → Files appear where your code expects them

## Full workflow example

```python linenums="1"
--8<-- "examples/04-catalog/catalog.py:29:52"
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
