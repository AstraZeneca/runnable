"""
This is a simple pipeline that does 3 steps in sequence.

    step 1 >> step 2 >> step 3 >> success

    All the steps are mocked and they will just pass through.
    Use this pattern to define the skeleton of your pipeline and flesh out the steps later.

    Note that you can give any arbitrary keys to the steps (like step 2). This is handy
    to mock steps within mature pipelines.

    You can run this pipeline by:
       python examples/mocking.py
"""

from runnable import Pipeline, Stub


def main():
    step1 = Stub(name="step1")  # (1)
    step2 = Stub(name="step2", what="is this thing")

    step3 = Stub(name="step3", terminate_with_success=True)  # (3)

    pipeline = Pipeline(steps=[step1, step2, step3], add_terminal_nodes=True)  # (4)

    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
