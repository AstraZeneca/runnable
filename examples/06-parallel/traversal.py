"""
This pipeline is same as the one seen in 02-sequential/traversal.py.

Given the naming convention used, we cannot import it directly.

"""

from examples.common.functions import hello
from runnable import NotebookTask, Pipeline, PythonTask, ShellTask, Stub


def main():
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

    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
