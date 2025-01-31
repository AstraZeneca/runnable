from examples.common.functions import hello, raise_ex
from runnable import NotebookTask, Parallel, Pipeline, PythonTask, ShellTask, Stub


def traversal_success():
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


def traversal_fail():
    """
    Use the pattern of using "execute" to control the execution of the pipeline.

    The same pipeline can be run independently from the command line.

    WARNING: If the execution is not controlled by "execute", the pipeline will be executed
    even during the definition of the branch in parallel steps.
    """
    stub_task = Stub(name="hello stub")

    python_task = PythonTask(
        name="hello python",
        function=raise_ex,
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
        branches={"branch1": traversal_success(), "branch2": traversal_fail()},
    )

    pipeline = Pipeline(steps=[parallel_step])

    pipeline.execute()
    return pipeline


if __name__ == "__main__":
    main()
