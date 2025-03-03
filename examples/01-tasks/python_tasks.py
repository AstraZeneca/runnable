"""
You can execute this pipeline by:

    python examples/01-tasks/python_tasks.py

The stdout of "Hello World!" would be captured as execution
log and stored in the catalog.
"""

from examples.common.functions import hello
from runnable import Pipeline, PythonTask


def main():
    # Create a tasks which calls the function "hello"
    # If this step executes successfully,
    # the pipeline will terminate with success
    hello_task = PythonTask(
        name="hello",
        function=hello,
        terminate_with_success=True,
    )

    # The pipeline has only one step.
    pipeline = Pipeline(steps=[hello_task])

    pipeline.execute()
    return pipeline


if __name__ == "__main__":
    main()
