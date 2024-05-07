"""
The below example shows how to set/get parameters in python
tasks of the pipeline.

The function, set_parameter, returns
    - JSON serializable types
    - pydantic models
    - pandas dataframe, any "object" type

pydantic models are implicitly handled by runnable
but "object" types should be marked as "pickled".

Use pickled even for python data types is advised for
reasonably large collections.

Run the below example as:
    python examples/03-parameters/passing_parameters_python.py

"""

from examples.common.functions import read_parameter, write_parameter
from runnable import Pipeline, PythonTask, metric, pickled


def main():
    write_parameters = PythonTask(
        function=write_parameter,
        returns=[
            pickled("df"),
            "integer",
            "floater",
            "stringer",
            "pydantic_param",
            metric("score"),
        ],
        name="set_parameter",
    )

    read_parameters = PythonTask(
        function=read_parameter,
        terminate_with_success=True,
        name="get_parameters",
    )

    pipeline = Pipeline(
        steps=[write_parameters, read_parameters],
    )

    _ = pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
