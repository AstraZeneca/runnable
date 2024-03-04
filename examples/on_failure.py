"""
This is a simple pipeline to demonstrate failure in a step.

    The default behavior is to traverse to step type fail and mark the run as failed.
    But you can control it by providing on_failure.

    In this example: step 1 fails and moves to step 3 skipping step 2. The pipeline status
    is considered to be success.

    step 1 (FAIL) >> step 3 >> success

    You can run this example by:
    python examples/on_failure.py
"""

from runnable import Pipeline, ShellTask, Stub


def main():
    step_1 = ShellTask(name="step 1", command="exit 1")
    step_2 = Stub(name="step 2")

    step_3 = Stub(name="step 3", terminate_with_success=True)

    step_1.on_failure = step_3.name

    step_1 >> step_2 >> step_3

    pipeline = Pipeline(
        steps=[step_1, step_2, step_3],
        start_at=step_1,
        add_terminal_nodes=True,
    )
    pipeline.execute()


if __name__ == "__main__":
    main()
