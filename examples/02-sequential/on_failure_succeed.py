"""
This pipeline showcases handling failures in a pipeline.

The path taken if none of the steps failed:
step_1 -> step_2 -> step_3 -> success

step_1 is a python function that raises an exception.
And we can instruct the pipeline to execute step_4 if step_1 fails
and then eventually succeed too.
step_1 -> step_4 -> success

This pattern is handy when you are expecting a failure of a step
and have ways to handle it.

Corresponds to:
try:
    step1()  # Raises the exception
    step2()
    step3()
except Exception as e:
    step4()

Run this pipeline:
    python examples/02-sequential/on_failure_succeed.py
"""

from examples.common.functions import raise_ex
from runnable import Pipeline, PythonTask, Stub


def main():
    step_1 = PythonTask(name="step_1", function=raise_ex)  # [concept:failing-task]

    step_2 = Stub(name="step_2")

    step_3 = Stub(name="step_3")

    on_failure_pipeline = Stub(name="step_4").as_pipeline()  # [concept:failure-pipeline]

    step_1.on_failure = on_failure_pipeline  # [concept:failure-handling]

    pipeline = Pipeline(  # [concept:pipeline]
        steps=[step_1, step_2, step_3],
    )
    pipeline.execute()  # [concept:execution]

    return pipeline


if __name__ == "__main__":
    main()
