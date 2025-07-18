"""
You can execute this pipeline by:

    python examples/01-tasks/scripts.py

The command can be anything that can be
executed in a shell.
The stdout/stderr of the execution is
captured as execution log and stored in the catalog.

"""

from runnable import Pipeline, ShellTask


def main():
    # If this step executes successfully, the pipeline will terminate with success
    hello_task = ShellTask(
        name="hello",
        command="echo 'Hello World!'",
    )

    # The pipeline has only one step.
    pipeline = Pipeline(steps=[hello_task])

    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
    main()
