"""
This is a simple pipeline that does 2 steps in sequence.
    In this example:
      1. First step: returns a "parameter" x as a Pydantic model
      2. Second step: Consumes that parameter and prints it

    This pipeline demonstrates one way to pass small data from one step to another.

    You can run this pipeline by: python examples/python-tasks.py
"""
from magnus import Pipeline, Task


def main():
    step1 = Task(
        name="step1",
        command="python --version",
        command_type="shell",
    )
    step2 = Task(
        name="step2",
        command="python --version",
        command_type="shell",
        terminate_with_success=True,
        overrides={"local-container": {"custom_docker_image": {}}},
    ).depends_on(step1)

    pipeline = Pipeline(
        start_at=step1,
        steps=[step1, step2],
        add_terminal_nodes=True,
    )

    pipeline.execute()


if __name__ == "__main__":
    main()
