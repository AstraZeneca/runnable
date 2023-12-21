"""
This is a stubbed pipeline that does 4 steps in sequence.
All the steps are mocked and they will just pass through.
Use this pattern to define the skeleton of your pipeline and
flesh out the steps later.

You can run this pipeline by python run examples/pipelines/traversal.py
"""

from magnus import Pipeline, Stub


def main():
    step_1 = Stub(name="Step 1")

    step_2 = Stub(name="Step 2")

    step_3 = Stub(name="Step 3", terminate_with_success=True)

    # link nodes
    step_1 >> step_2 >> step_3

    """
            or
    step_1 << step_2 << step_3
            or

    step_2.depends_on(step_1)
    step_3.depends_on(step_2)
    """

    pipeline = Pipeline(
        steps=[step_1, step_2, step_3],
        start_at=step_1,
        add_terminal_nodes=True,
    )

    run_log = pipeline.execute()
    print(run_log)


if __name__ == "__main__":
    main()
