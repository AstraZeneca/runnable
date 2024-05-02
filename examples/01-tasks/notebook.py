"""
You can execute this pipeline by:

    python examples/01-tasks/notebook.py

The notebook is executed in the same environment so any installed packages are available for the
notebook.

Upon successful execution, the output notebook with cell outputs is stored in the catalog.
For example, the catalog structure for this execution would be:

.catalog
└── meek-rosalind-0853
    ├── examples
    │   └── common
    │       └── simple_notebook_out.ipynb
    └── notebook.execution.log

The notebook simple_notebook_<step name>_out.ipynb has the captured stdout of "Hello World!".
"""

from runnable import NotebookTask, Pipeline


def main():
    # Execute the notebook present in examples/common/simple_notebook.ipynb.
    # The path is relative to the project root.
    # If this step executes successfully, the pipeline will terminate with success
    hello_task = NotebookTask(
        name="hello",
        notebook="examples/common/simple_notebook.ipynb",
        terminate_with_success=True,
    )

    # The pipeline has only one step.
    pipeline = Pipeline(steps=[hello_task])

    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
