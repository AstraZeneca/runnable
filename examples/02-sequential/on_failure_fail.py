"""
This pipeline showcases handling failures in a pipeline.

The path taken if none of the steps failed:
step_1 -> step_2 -> step_3 -> success

step_1 is a python function that raises an exception.
And we can instruct the pipeline to execute step_4 if step_1 fails
and then eventually fail.
step_1 -> step_4 -> fail

This pattern is handy when you need to do something before eventually
failing (eg: sending a notification, updating status, etc...)

Run this pipeline as:
    python examples/02-sequential/on_failure_fail.py
"""

from examples.common.functions import raise_ex
from runnable import Pipeline, PythonTask, Stub


def main():
    step_1 = PythonTask(name="step 1", function=raise_ex)  # This will fail

    step_2 = Stub(name="step 2")

    step_3 = Stub(name="step 3", terminate_with_success=True)
    step_4 = Stub(name="step 4", terminate_with_failure=True)  # (1)

    step_1.on_failure = step_4.name

    pipeline = Pipeline(
        steps=[step_1, step_2, step_3],
    )
    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
