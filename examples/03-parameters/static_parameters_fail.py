"""
The below example showcases setting up known initial parameters for a pipeline
of only python tasks

The initial parameters as defined in the yaml file are:
    simple: 1
    complex_param:
        x: 10
        y: "hello world!!"

runnable allows using pydantic models for deeply nested parameters and
casts appropriately based on annotation. eg: read_initial_params_as_pydantic

If no annotation is provided, the parameter is assumed to be a dictionary.
eg: read_initial_params_as_json

You can set the initial parameters from environment variables as well.
eg: Any environment variable prefixed by "RUNNABLE_PRM_" will be picked up by runnable

Run this pipeline as:
    python examples/03-parameters/static_parameters_python.py

"""

import os

from examples.common.functions import raise_ex
from runnable import NotebookTask, Pipeline, PythonTask


def main():
    read_params_in_notebook = NotebookTask(
        name="read_params_in_notebook",
        notebook="examples/common/read_parameters.ipynb",
        terminate_with_success=True,
    )

    notebook_pipeline = Pipeline(
        steps=[
            read_params_in_notebook,
        ],
    )
    read_params_and_fail = PythonTask(
        function=raise_ex,
        name="read_params_and_fail",
        terminate_with_success=True,
    )

    read_params_and_fail.on_failure = notebook_pipeline

    python_pipeline = Pipeline(
        steps=[
            read_params_and_fail,
        ],
    )

    python_pipeline.execute(parameters_file="examples/common/initial_parameters.yaml")

    return python_pipeline


if __name__ == "__main__":
    # Any parameter prefixed by "RUNNABLE_PRM_" will be picked up by runnable
    os.environ["RUNNABLE_PRM_envvar"] = "from env"
    main()
