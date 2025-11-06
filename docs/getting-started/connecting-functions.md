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

# Step 1: Create data with named outputs
step1 = PythonTask(
    function=write_parameter,
    returns=[pickled("df"), "integer", "floater", "stringer", "pydantic_param", metric("score")]
)

# Step 2: Process data - parameters matched automatically!
step2 = PythonTask(function=read_parameter)

pipeline = Pipeline(steps=[step1, step2])
pipeline.execute()
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

pipeline = Pipeline(steps=[
    PythonTask(function=create_data, returns=[pickled("df")]),
    NotebookTask(notebook_path="process.ipynb", returns=["processed_df"]),
    ShellTask(command="./analyze.sh", returns=["report_path"]),
    PythonTask(function=send_email)  # Gets report_path automatically
])
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

!!! info "Universal pattern"

    This automatic parameter passing works seamlessly between **any** task types:

    ```python
    Pipeline(steps=[
        PythonTask(function=extract_data, returns=["raw_df"]),      # Python →
        NotebookTask(notebook_path="clean.ipynb", returns=["df"]), # → Notebook →
        ShellTask(command="./analyze.sh", returns=["report"])      # → Shell
    ])
    ```

    Whether it's Python → Python, Notebook → Shell, or any combination - parameters flow automatically!

Next: Understand [when to use jobs vs pipelines](../concepts/building-blocks/jobs-vs-pipelines.md).
