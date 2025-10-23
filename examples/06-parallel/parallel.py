"""
You can execute this pipeline by:

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
    parallel_step = Parallel(  # [concept:parallel]
        name="parallel_step",
        branches={
            "branch1": traversal(),  # [concept:branch-definition]
            "branch2": traversal(),  # [concept:branch-definition]
        },
    )

    continue_to = Stub(name="continue to")  # [concept:continuation]

    pipeline = Pipeline(steps=[parallel_step, continue_to])  # [concept:pipeline]

    pipeline.execute()  # [concept:execution]
    return pipeline


if __name__ == "__main__":
    main()
