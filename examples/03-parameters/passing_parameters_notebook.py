"""
Demonstrates passing parameters to and from a notebook.

We can extract json, pydantic, objects from notebook.
eg: write_parameters_from_notebook

But can only inject json type parameters to a notebook.
eg: read_parameters_in_notebook
pydantic parameters are injected as dict.

"""

from examples.common.functions import read_parameter
from runnable import NotebookTask, Pipeline, PythonTask, metric, pickled


def main():
    write_parameters_from_notebook = NotebookTask(
        notebook="examples/common/write_parameters.ipynb",
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
        name="get_parameters",
    )

    read_parameters_in_notebook = NotebookTask(
        notebook="examples/common/read_parameters.ipynb",
        terminate_with_success=True,
        name="read_parameters_in_notebook",
    )

    pipeline = Pipeline(
        steps=[write_parameters_from_notebook, read_parameters, read_parameters_in_notebook],
    )

    _ = pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
