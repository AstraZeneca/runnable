"""
This example demonstrates the use of the Parallel step.

The branches of the parallel step are themselves pipelines and can be defined
as shown in 02-sequential/traversal.py.

WARNING, the function returning the pipeline should not executed
during the definition of the branch in parallel steps.

Run this pipeline as:
    python examples/06-parallel/parallel.py
"""

from examples.common.functions import hello
from runnable import NotebookTask, Parallel, Pipeline, PythonTask, ShellTask, Stub


def traversal():
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

    return pipeline


def main():
    parallel_step = Parallel(
        name="parallel_step",
        terminate_with_success=True,
        branches={"branch1": traversal(), "branch2": traversal()},
    )

    pipeline = Pipeline(steps=[parallel_step])

    pipeline.execute()
    return pipeline


if __name__ == "__main__":
    main()
