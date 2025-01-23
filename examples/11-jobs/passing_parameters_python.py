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

from examples.common.functions import write_parameter
from runnable import PythonJob, metric, pickled


def main():
    job = PythonJob(
        function=write_parameter,
        returns=[
            pickled("df"),
            "integer",
            "floater",
            "stringer",
            "pydantic_param",
            metric("score"),
        ],
    )

    job.execute()

    return job


if __name__ == "__main__":
    main()
