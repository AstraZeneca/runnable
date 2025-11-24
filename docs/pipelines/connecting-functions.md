# 🔗 Connecting Functions

The magic happens when you chain functions together. Runnable makes this effortless.

## What you already know

You probably chain functions like this:

```python
def write_parameter():
    df = pd.DataFrame({"x": [1, 2, 3]})
    return df, 10, 3.14, "hello", SamplePydanticModel(x=10, foo="bar"), 0.95

def read_parameter(df, integer, floater, stringer, pydantic_param, score):
    print(f"Received: df={len(df)} rows, integer={integer}, score={score}")
    return df.mean()

# Manual chaining
df, integer, floater, stringer, pydantic_param, score = write_parameter()
result = read_parameter(df, integer, floater, stringer, pydantic_param, score)
```

## Runnable does the chaining for you

Same functions, automatic parameter passing:

```python
from runnable import Pipeline, PythonTask, pickled, metric

def main():
    # Step 1: Create data with named outputs
    step1 = PythonTask(
        function=write_parameter,
        returns=[pickled("df"), "integer", "floater", "stringer", "pydantic_param", metric("score")]
    )

    # Step 2: Process data - parameters matched automatically!
    step2 = PythonTask(function=read_parameter)

    pipeline = Pipeline(steps=[step1, step2])
    pipeline.execute()
    return pipeline

if __name__ == "__main__":
    main()
```

✨ **Magic**: The `df` returned by `write_parameter` automatically becomes the `df` parameter for `read_parameter`.

??? example "See complete runnable code"
    ```python title="examples/03-parameters/passing_parameters_python.py"
    --8<-- "examples/03-parameters/passing_parameters_python.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/03-parameters/passing_parameters_python.py
    ```

## How it works

1. **Step 1** returns values with names: `returns=["df", "score"]`
2. **Step 2** function signature: `def analyze(df, score):`
3. **Runnable matches** return names to parameter names automatically

## Mix different task types

Python functions, notebooks, and shell scripts all work together:

```python
from runnable import Pipeline, PythonTask, NotebookTask, ShellTask

def main():
    pipeline = Pipeline(steps=[
        PythonTask(function=create_data, returns=[pickled("df")]),
        NotebookTask(notebook_path="process.ipynb", returns=["processed_df"]),
        ShellTask(command="./analyze.sh", returns=["report_path"]),
        PythonTask(function=send_email)  # Gets report_path automatically
    ])
    pipeline.execute()
    return pipeline

if __name__ == "__main__":
    main()
```

??? example "See complete runnable code"
    ```python title="examples/02-sequential/traversal.py"
    --8<-- "examples/02-sequential/traversal.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/02-sequential/traversal.py
    ```

!!! tip "Parameter matching"

    Return names must match parameter names. `returns=["data"]` → `def process(data):`

!!! info "Parameter Type Compatibility"

    Parameter passing works between task types, but with important constraints based on data types and how each task type receives parameters:

    **How Parameters Are Passed:**

    | Task Type | How Parameters Are Received | Input Parameters | Output Parameters |
    |-----------|----------------------------|------------------|------------------|
    | **Python** | Function arguments | All types (primitive, pickled, pydantic, metric) | All types (primitive, pickled, pydantic, metric) |
    | **Notebook** | Tagged parameter cells (variables replaced) | Python primitives only (int, str, float, list, dict) | All types (primitive, pickled, pydantic, metric) |
    | **Shell** | Environment variables | Python primitives only (int, str, float, list, dict) | Python primitives only (int, str, float, list, dict) |

    **Notebook Parameter Mechanism:**

    Notebooks receive parameters through tagged cells where variable values are replaced:

    ```python
    # In your notebook's first cell (tagged as "parameters"):
    count = None      # This will be replaced with actual value
    status = None     # This will be replaced with actual value
    ```

    **✅ This works:**
    ```python
    def main():
        Pipeline(steps=[
            PythonTask(function=extract_data, returns=["count", "status"]),    # primitives →
            NotebookTask(notebook="clean.ipynb", returns=["df"]),              # → notebook receives via parameter cells
            PythonTask(function=analyze, returns=["report"])                   # → python can receive pickled df
        ]).execute()

    if __name__ == "__main__":
        main()
    ```

    **❌ This won't work:**
    ```python
    def main():
        Pipeline(steps=[
            PythonTask(function=create_model, returns=[pickled("model")]),     # pickled object →
            NotebookTask(notebook="use_model.ipynb")                           # → notebook can't receive pickled objects!
        ]).execute()

    if __name__ == "__main__":
        main()
    ```

Next: Understand [when to use jobs vs pipelines](jobs-vs-pipelines.md).
