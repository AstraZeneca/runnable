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

from examples.common.functions import (
    read_initial_params_as_json,
    read_initial_params_as_pydantic,
)
from runnable import Pipeline, PythonTask


def main():
    """
    Signature of read_initial_params_as_pydantic
    def read_initial_params_as_pydantic(
        integer: int,
        floater: float,
        stringer: str,
        pydantic_param: ComplexParams,
        envvar: str,
    ):
    """
    read_params_as_pydantic = PythonTask(
        function=read_initial_params_as_pydantic,
        name="read_params_as_pydantic",
    )

    """
    Signature of read_initial_params_as_json
    def read_initial_params_as_json(
        integer: int,
        floater: float,
        stringer: str,
        pydantic_param: Dict[str, Union[int, str]],
    ):
    """
    read_params_as_json = PythonTask(
        function=read_initial_params_as_json,
        terminate_with_success=True,
        name="read_params_json",
    )

    pipeline = Pipeline(
        steps=[read_params_as_pydantic, read_params_as_json],
    )

    _ = pipeline.execute(parameters_file="examples/common/initial_parameters.yaml")

    return pipeline


if __name__ == "__main__":
    # Any parameter prefixed by "RUNNABLE_PRM_" will be picked up by runnable
    os.environ["RUNNABLE_PRM_envvar"] = "from env"
    main()
