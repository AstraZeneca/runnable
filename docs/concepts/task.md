Task nodes are the execution units of the pipeline.

They can be [python functions](#python_functions), [notebooks](#notebook),
[shell scripts](#shell) or [stubs](#stub)

In the below examples, highlighted lines of the code are the relevant bits while
the rest of the python code defines and executes a pipeline that executes
the python function/notebook/shell script/stubs.


---

## Python functions

Uses python functions as tasks.

[API Documentation](../reference.md/#pythontask)

### Example

!!! tip inline end "Structuring"

    It is best to keep the application specific functions in a different module
    than the pipeline definition, if you are using Python SDK.

```python linenums="1"
--8<-- "examples/01-tasks/python_tasks.py"
```

Key concepts in this example:

- `[concept:task]`: Creating a PythonTask that wraps your function
- `[concept:pipeline]`: Building a pipeline with the task
- `[concept:execution]`: Running the pipeline

<!-- Please refer to [field reference](../sdk.md/#runnable.PythonTask). -->

<hr style="border:2px dotted orange">

## Notebook


Jupyter notebooks are supported as tasks. We internally use
[Ploomber engine](https://github.com/ploomber/ploomber-engine) for executing notebooks.

The output is saved to the same location as the input notebook but with ```_out``` post-fixed to
the name of the notebook and is also saved in the ```catalog``` for logging and ease of debugging.

[API Documentation](../reference.md/#notebooktask)

### Example

```python linenums="1"
--8<-- "examples/01-tasks/notebook.py"
```

Key concepts in this example:

- `[concept:notebook-task]`: Creating a NotebookTask that executes a Jupyter notebook
- `[concept:pipeline]`: Building a pipeline with the task
- `[concept:execution]`: Running the pipeline

<hr style="border:2px dotted orange">

## Shell

[Python functions](#python_functions) and [Jupyter notebooks](#notebook) provide a rich interface to the python
ecosystem while shell provides a interface to non-python executables.

[API Documentation](../reference.md/#shelltask)

### Example

```python linenums="1"
--8<-- "examples/01-tasks/scripts.py"
```

Key concepts in this example:

- `[concept:shell-task]`: Creating a ShellTask that executes shell commands
- `[concept:pipeline]`: Building a pipeline with the task
- `[concept:execution]`: Running the pipeline


## Stub

Stub nodes in runnable are just like ```pass``` or ```...``` in python code.
It is a placeholder and useful when you want to debug ordesign your pipeline.

Stub nodes can take arbitrary number of parameters and is always a success.

[API Documentation](../reference.md/#stub)

### Example

!!! note annotate inline end "Intuition"

    Designing a pipeline is similar to writing a modular program. Stub nodes are handy to create a placeholder
    for some step that will be implemented in the future.

    During debugging, changing a node to ```stub``` will let you focus on the actual bug without having to
    execute the additional steps.

```python linenums="1"
--8<-- "examples/01-tasks/stub.py"
```

Key concepts in this example:

- `[concept:stub-task]`: Creating placeholder tasks that always succeed
- `[concept:stub-with-params]`: Stub tasks can take arbitrary parameters for flexibility
- `[concept:pipeline]`: Building a pipeline with multiple tasks in sequence
- `[concept:execution]`: Running the pipeline
