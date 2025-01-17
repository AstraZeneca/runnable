"""
You can execute this pipeline by:

    python examples/01-tasks/scripts.py

The command can be anything that can be executed in a shell.
The stdout/stderr of the execution is captured as execution log and stored in the catalog.

For example:

.catalog
└── seasoned-perlman-1355
    └── hello.execution.log

"""

from runnable import Job, ShellTask


def main():
    # If this step executes successfully, the pipeline will terminate with success
    hello_task = ShellTask(
        name="hello",
        command="echo 'Hello World!'",
        terminate_with_success=True,
    )

    job = Job(name="hello", task=hello_task)

    job.execute()

    return job


if __name__ == "__main__":
    main()
