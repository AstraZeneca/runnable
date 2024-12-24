"""
Example to show case nesting of parallel steps.

runnable does not put a limit on the nesting of parallel steps.
Deeply nested pipelines can be hard to read and not all
executors support it.

Run this pipeline as:
    python examples/06-parallel/nesting.py
"""

from examples.common.functions import hello
from runnable import NotebookTask, Parallel, Pipeline, PythonTask, ShellTask, Stub


def traversal(execute: bool = True):
    """
    Use the pattern of using "execute" to control the execution of the pipeline.

    The same pipeline can be run independently from the command line.

    WARNING: If the execution is not controlled by "execute", the pipeline will be executed
    even during the definition of the branch in parallel steps.
    """
    stub_task = Stub(name="hello stub")

    python_task = PythonTask(
        name="hello python",
        function=hello,
    )

    shell_task = ShellTask(
        name="hello shell",
        command="echo 'Hello World!'",
    )

    notebook_task = NotebookTask(
        name="hello notebook",
        notebook="examples/common/simple_notebook.ipynb",
        terminate_with_success=True,
    )

    # The pipeline has a mix of tasks.
    # The order of execution follows the order of the tasks in the list.
    pipeline = Pipeline(steps=[stub_task, python_task, shell_task, notebook_task])

    if execute:  # Do not execute the pipeline if we are using it as a branch
        pipeline.execute()

    return pipeline


def parallel_pipeline(execute: bool = True):
    parallel_step = Parallel(
        name="parallel step",
        terminate_with_success=True,
        branches={
            "branch1": traversal(execute=False),
            "branch2": traversal(execute=False),
        },
    )

    pipeline = Pipeline(steps=[parallel_step])

    if execute:
        pipeline.execute()
    return pipeline


def main():
    # Create a parallel step with parallel steps as branches.
    parallel_step = Parallel(
        name="nested_parallel",
        terminate_with_success=True,
        branches={
            "branch1": parallel_pipeline(execute=False),
            "branch2": parallel_pipeline(execute=False),
        },
    )

    pipeline = Pipeline(steps=[parallel_step])
    pipeline.execute()
    return pipeline


if __name__ == "__main__":
    main()
