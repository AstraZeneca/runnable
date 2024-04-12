Examples in this section are ordered from simple to advanced.
All examples have both python SDK and yaml representations.

Please use this as an index to find specific example.


- common: Has python functions/notebooks/scripts that are used across the examples

- 01-tasks: Examples of the tasks that can be part of the pipeline.

    - [stub.py](./01-tasks/stub.py), [stub.yaml](./01-tasks/stub.yaml): demonstrates the concept of a stub.
    
    - [python_tasks.py](./01-tasks/python_tasks.py), [python_tasks.yaml](./01-tasks/python_tasks.yaml): uses python functions as tasks.
        The stdout/stderr of all the tasks are captured and stored in the catalog.
    - [notebook.py](./01-tasks/notebook.py), [notebook.yaml](./01-tasks/notebook.yaml): uses notebooks as tasks
        The executed notebook is captured in the catalog.
    - [scripts.py](./01-tasks/scripts.py), [scripts.yaml](./01-tasks/scripts.yaml): uses shell scripts as tasks
        The stdout/stderr of all scripts are captured and stored in the catalog.


The above examples showcase executable units of the pipeline.
The next section has examples on stitching these tasks together for complex operations.

- 02-sequential: Examples of stitching tasks together including behavior in case of failures.

    - traversal: A pipeline which is a mixed bag of notebooks, python functions and 
    shell scripts.
    - default_failure: The default failure behavior.
    - on_failure_fail: On failure of a step, do some action and fail
    - on_failure_success: On failure of a step, take a different route


The above examples show stitching complex operations of the pipeline.
The next section has examples on 

- 03: Examples of passing parameters between tasks


