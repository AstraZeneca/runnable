"""
You can execute this pipeline by:

    python examples/01-tasks/python_tasks.py

The stdout of "Hello World!" would be captured as execution
log and stored in the catalog.
"""

from examples.common.functions import hello
from runnable import PythonTask


def main():
    # A single step pipeline can also be created using the
    # task.as_pipeline() method.
    pipeline = PythonTask(
        name="hello",
        function=hello,
    ).as_pipeline()

    pipeline.execute()
    return pipeline


if __name__ == "__main__":
    main()
