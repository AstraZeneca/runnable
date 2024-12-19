"""
When defining a Pipeline(), it automatically adds a success node and failure node.

By default any failure in a step is considered to be a failure in the pipeline.

In the below example, the progression would be as follows:

  step 1 >> step 2 >> fail


You can run this example by:
  python examples/02-sequential/default_fail.py
"""

from examples.common.functions import hello, raise_ex
from runnable import Pipeline, PythonTask, Stub


def main():
    step1 = PythonTask(name="step 1", function=hello)

    step2 = PythonTask(name="step 2", function=raise_ex)  # This will fail

    step3 = Stub(
        name="step 3", terminate_with_success=True
    )  # This step will not be executed

    pipeline = Pipeline(steps=[step1, step2, step3])

    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
