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

    - [traversal.py](./02-sequential/traversal.py), [traversal.yaml](./02-sequential/traversal.yaml): A pipeline which is a mixed bag of notebooks, python functions and
    shell scripts.
    - [default_fail.py](./02-sequential/default_fail.py), [default_fail.yaml](./02-sequential/default_fail.yaml): The default failure behavior.
    - [on_failure_fail](./02-sequential/on_failure_fail.py), [on_faliure_fail.yaml](./02-sequential/on_failure_fail.yaml) On failure of a step, do some action and fail
    - [on_failure_success.py](./02-sequential/on_failure_succeed.py), [on_failure_success.yaml](./02-sequential/on_failure_succeed.yaml): On failure of a step, take a different route


The above examples show stitching complex operations of the pipeline.
The next section has examples on communicating between tasks during execution.

- 03: Examples of passing parameters between tasks of a pipeline.

    Guidelines:

        - python functions can get/set simple python data types, pydantic models, objects marked as pickled. Some of the
        simple data types can also be marked as a metric.
        -


    - [static_parameters_python.py](./03-parameters/static_parameters_python.py), [static_parameters_python.yaml](./03-parameters/static_parameters_python.yaml): A pipeline to show the access of static or known parameters by python tasks.

    - [static_parameters_non_python.py](./03-parameters/static_parameters_non_python.py), [static_parameters_non_python.yaml](./03-parameters/static_parameters_non_python.yaml): A pipeline to show the access of static or known parameters by python tasks.

    - [passing_parameters_python.py](./03-parameters/passing_parameters_python.py), [passing_parameters_python.yaml](./03-parameters/passing_parameters_python.yaml): shows the mechanism of passing parameters (simple python datatypes, "dillable" objects, pydantic models) and registering metrics between python tasks.

    - [passing_parameters_notebook.py](./03-parameters/passing_parameters_notebook.py), [passing_parameters_notebook.yaml](./03-parameters/passing_parameters_notebook.yaml): shows the mechanism of passing parameters (simple python datatypes, "dillable" objects, pydantic models) and registering metrics between tasks. runnable can "get" object
    parameters from notebooks but cannot inject them into notebooks.
