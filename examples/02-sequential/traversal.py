"""
You can execute this pipeline by:

    python examples/02-sequential/traversal.py

    A pipeline can have any "tasks" as part of it. In the
    below example, we have a mix of stub, python, shell and notebook tasks.

    As with simpler tasks, the stdout and stderr of each task are captured
    and stored in the catalog.

    .catalog
    └── cold-jennings-1534
        ├── examples
        │   └── common
        │       └── simple_notebook_out.ipynb
        ├── hello_notebook.execution.log
        ├── hello_python.execution.log
        └── hello_shell.execution.log

    4 directories, 4 files

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
    pipeline = Pipeline(
        steps=[  # (2)
            stub_task,  # (1)
            python_task,
            shell_task,
            notebook_task,
        ]
    )

    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
