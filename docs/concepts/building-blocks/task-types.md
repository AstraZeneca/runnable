# 🛠️ Different Types of Tasks

Runnable works with Python functions, Jupyter notebooks, and shell scripts. Use whatever fits your workflow.

## 🐍 Python functions

Your regular functions work as-is:

```python
from runnable import Pipeline, PythonTask
from examples.common.functions import hello

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

Perfect for: Data processing, ML models, business logic

## 📓 Jupyter notebooks

Run notebooks as pipeline steps:

```python
from runnable import Pipeline, NotebookTask

task = NotebookTask(
    name="analyze",
    notebook="examples/common/simple_notebook.ipynb"
)
pipeline = Pipeline(steps=[task])
pipeline.execute()
```

??? example "See complete runnable code"
    ```python title="examples/01-tasks/notebook.py"
    --8<-- "examples/01-tasks/notebook.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/01-tasks/notebook.py
    ```

Perfect for: Exploration, visualization, reporting

## 🔧 Shell commands

Run any command-line tool:

```python
from runnable import Pipeline, ShellTask

task = ShellTask(
    name="greet",
    command="echo 'Hello World!'"
)
pipeline = Pipeline(steps=[task])
pipeline.execute()
```

??? example "See complete runnable code"
    ```python title="examples/01-tasks/scripts.py"
    --8<-- "examples/01-tasks/scripts.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/01-tasks/scripts.py
    ```

Perfect for: System commands, external tools, legacy scripts

## 🎭 Mock tasks for testing

Use stubs when building workflows:

```python
from runnable import Pipeline, Stub

# Create placeholder steps that always succeed
step1 = Stub(name="extract_data")
step2 = Stub(name="process_data", what="placeholder")
step3 = Stub(name="save_results")

pipeline = Pipeline(steps=[step1, step2, step3])
pipeline.execute()
```

??? example "See complete runnable code"
    ```python title="examples/01-tasks/stub.py"
    --8<-- "examples/01-tasks/stub.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/01-tasks/stub.py
    ```

Perfect for: Testing pipeline structure, placeholder steps

## 🔄 Mix and match

Same workflow, different tools:

```python
from runnable import Pipeline, PythonTask, NotebookTask, ShellTask

pipeline = Pipeline(steps=[
    PythonTask(function=extract_data, returns=["raw_df"]),      # Python function
    NotebookTask(notebook="clean.ipynb", returns=["clean_df"]), # Jupyter notebook
    ShellTask(command="./analyze.sh", returns=["report_path"])  # Shell script
])
pipeline.execute()
```

??? example "See complete runnable code"
    ```python title="examples/02-sequential/traversal.py"
    --8<-- "examples/02-sequential/traversal.py"
    ```

    **Try it now:**
    ```bash
    uv run examples/02-sequential/traversal.py
    ```

Each task type can:

- ✅ Accept parameters from previous steps
- ✅ Return data to next steps
- ✅ Use the same configuration system
- ✅ Run on any environment (local, container, Kubernetes)

!!! tip "Choose the right tool"

    - **Python**: Fast, type-safe, great for algorithms
    - **Notebooks**: Great for exploration and reports with visualizations
    - **Shell**: Perfect for calling existing tools or system commands
    - **Stubs**: Useful for testing workflow structure

Next: Learn how to add [external configuration](../superpowers/parameters-from-outside.md) without changing your code.
