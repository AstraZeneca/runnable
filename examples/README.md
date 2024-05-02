Examples in this section are ordered from simple to advanced.
All examples have both python SDK and yaml representations.

Please use this as an index to find specific example.


- [common](./common/): Has python functions/notebooks/scripts that are used across the examples

- 01-tasks: Examples of the tasks that can be part of the pipeline.

    - [stub.py](./01-tasks/stub.py), [stub.yaml](./01-tasks/stub.yaml): demonstrates the concept of a stub.

    - [python_tasks.py](./01-tasks/python_tasks.py), [python_tasks.yaml](./01-tasks/python_tasks.yaml): uses python functions as tasks.
        The stdout/stderr of all the tasks are captured and stored in the catalog.

    - [notebook.py](./01-tasks/notebook.py), [notebook.yaml](./01-tasks/notebook.yaml): uses notebooks as tasks
        The executed notebook is captured in the catalog.

    - [scripts.py](./01-tasks/scripts.py), [scripts.yaml](./01-tasks/scripts.yaml): uses shell scripts as tasks
        The stdout/stderr of all scripts are captured and stored in the catalog.

---


This section has examples on stitching these tasks together for complex operations.
We only show sequential pipeline while parallel and dynamic pipelines are
shown in later sections.

- 02-sequential: Examples of stitching tasks together including behavior in case of failures.

    - [traversal.py](./02-sequential/traversal.py), [traversal.yaml](./02-sequential/traversal.yaml): A pipeline which is a mixed bag of notebooks, python functions and shell scripts.

    - [default_fail.py](./02-sequential/default_fail.py), [default_fail.yaml](./02-sequential/default_fail.yaml): The default failure behavior.

    - [on_failure_fail](./02-sequential/on_failure_fail.py), [on_faliure_fail.yaml](./02-sequential/on_failure_fail.yaml) On failure of a step, do some action and fail

    - [on_failure_success.py](./02-sequential/on_failure_succeed.py), [on_failure_success.yaml](./02-sequential/on_failure_succeed.yaml): On failure of a step, take a different route and succeed


---

This section has examples on communicating between tasks during execution.
We only focusses on "parameters" while the next section focusses on "files".

- 03: Examples of passing parameters between tasks of a pipeline.

    Below table summarizes the input/output types of different task types. For ex: notebooks can only take JSON serializable
    parameters as input but can return json/pydantic/objects. Any python object that could be serialized using "dill" can be used.

    |          | Input                    | Output                   |
    | -------- | :---------------------:  | :----------------------: |
    | python   | json, pydantic, object   | json, pydantic, object   |
    | notebook | json                     | json, pydantic, object   |
    | shell    | json                     | json                     |


    - [static_parameters_python.py](./03-parameters/static_parameters_python.py), [static_parameters_python.yaml](./03-parameters/static_parameters_python.yaml): A pipeline to show the access of static or known parameters by python tasks.

        Any environment variables prefixed by RUNNABLE_PRM_ are recognized as parameters and
        can override parameters defined by the file.

    - [static_parameters_non_python.py](./03-parameters/static_parameters_non_python.py), [static_parameters_non_python.yaml](./03-parameters/static_parameters_non_python.yaml): A pipeline to show the access of static or known parameters by python tasks.

        Any environment variables prefixed by RUNNABLE_PRM_ are recognized as parameters and
        can override parameters defined by the file.

    - [passing_parameters_python.py](./03-parameters/passing_parameters_python.py), [passing_parameters_python.yaml](./03-parameters/passing_parameters_python.yaml): shows the mechanism of passing parameters (JSON serializable, objects, pydantic models) and registering metrics between python tasks.

    - [passing_parameters_notebook.py](./03-parameters/passing_parameters_notebook.py), [passing_parameters_notebook.yaml](./03-parameters/passing_parameters_notebook.yaml): shows the mechanism of passing parameters between notebook tasks. Please note that
    we cannot inject pydantic models or objects into the notebook but can capture them as return values.

    - [passing_parameters_shell.py](./03-parameters/passing_parameters_shell.py), [passing_parameters_shell.yaml](./03-parameters/passing_parameters_shell.yaml): shows the mechanism of passing parameters between shell tasks. Please note that
    we cannot inject/capture pydantic models or objects in shells.

---

This section focusses on moving files between tasks.

- 04: Examples of moving files between tasks of the pipeline.

    - [catalog.py](./04-catalog/catalog.py), [catalog.yaml](./04-catalog/catalog.yaml): demonstrate moving files between python, shell and notebook tasks.

---

This section focusses on exposing secrets to tasks. All secrets are exposed as environment
variables. The secrets are destroyed after the completion of the task.


---

Below are the examples of constructing parallel graphs and nested graphs.

Creating parallel graphs is simple as the branches are themselves pipelines.
