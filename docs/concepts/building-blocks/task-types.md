# 🛠️ Different Types of Tasks

Runnable works with Python functions, Jupyter notebooks, and shell scripts. Use whatever fits your workflow.

## 🐍 Python functions

Your regular functions work as-is:

```python linenums="1"
--8<-- "examples/01-tasks/python_tasks.py:7:17"
```

Perfect for: Data processing, ML models, business logic

## 📓 Jupyter notebooks

Run notebooks as pipeline steps:

```python linenums="1"
--8<-- "examples/01-tasks/notebook.py:7:15"
```

Perfect for: Exploration, visualization, reporting

## 🔧 Shell commands

Run any command-line tool:

```python linenums="1"
--8<-- "examples/01-tasks/scripts.py:7:15"
```

Perfect for: System commands, external tools, legacy scripts

## 🎭 Mock tasks for testing

Use stubs when building workflows:

```python linenums="1"
--8<-- "examples/01-tasks/stub.py:7:14"
```

Perfect for: Testing pipeline structure, placeholder steps

## 🔄 Mix and match

Same workflow, different tools:

```python linenums="1"
--8<-- "examples/02-sequential/traversal.py:13:25"
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
