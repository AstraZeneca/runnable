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

from runnable import Pipeline, Task


def main():
    step1 = Task(
        name="step1",  # (1)
        command="examples.functions.return_parameter",  # (2)
    )

    step2 = Task(
        name="step2",
        command="examples.functions.display_parameter",
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


if __name__ == "__main__":
    main()
