"""
You can execute this pipeline by:

    python examples/01-tasks/notebook.py

The notebook is executed in the same environment
so any installed packages are available for the
notebook.

Upon successful execution, the output notebook with
cell outputs is stored in the catalog.

"""

from runnable import NotebookTask, Pipeline


def main():
    # Execute the notebook present in examples/common/simple_notebook.ipynb.
    # The path is relative to the project root.
    # If this step executes successfully, the pipeline will terminate with success
    hello_task = NotebookTask(  # [concept:notebook-task]
        name="hello",
        notebook="examples/common/simple_notebook.ipynb",
    )

    # The pipeline has only one step.
    pipeline = Pipeline(steps=[hello_task])  # [concept:pipeline]

    pipeline.execute()  # [concept:execution]

    return pipeline


if __name__ == "__main__":
    main()
