"""
This is a simple pipeline that does 2 steps
in sequence.

In this example:
    1. First step: returns a "parameter" x as
        a Pydantic model
    2. Second step: Consumes that parameter and
        prints it

    You can run this pipeline by: python examples/python-tasks.py
"""

from examples.functions import display_parameter, return_parameter
from runnable import Pipeline, PythonTask


def main():
    step1 = PythonTask(
        name="step1",  # (1)
        function=return_parameter,  # (2)
    )

    step2 = PythonTask(
        name="step2",
        function=display_parameter,
        terminate_with_success=True,  # (3)
    )

    step1 >> step2  # (4)

    pipeline = Pipeline(
        start_at=step1,  # (5)
        steps=[step1, step2],  # (6)
        add_terminal_nodes=True,  # (7)
    )

    run_log = pipeline.execute()  # (8)
    print(run_log)

    return pipeline


if __name__ == "__main__":
    main()
