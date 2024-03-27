"""
This is a stubbed pipeline that does 4 steps in sequence.
All the steps are mocked and they will just pass through.
Use this pattern to define the skeleton of your pipeline and
flesh out the steps later.

You can run this pipeline by python run examples/pipelines/traversal.py
"""

from runnable import Pipeline, Stub


def main():
    step_1 = Stub(name="Step 1")

    step_2 = Stub(name="Step 2")

    step_3 = Stub(name="Step 3", terminate_with_success=True)

    pipeline = Pipeline(
        steps=[step_1, step_2, step_3],
        add_terminal_nodes=True,
    )

    run_log = pipeline.execute()
    print(run_log)

    return pipeline


if __name__ == "__main__":
    main()
