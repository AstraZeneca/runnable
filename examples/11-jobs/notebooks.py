"""
You can execute this pipeline by:

    python examples/11-jobs/notebooks.py

The output of the notebook will be captured as execution
log and stored in the catalog.


"""

from runnable import NotebookJob


def main():
    job = NotebookJob(
        notebook="examples/common/simple_notebook.ipynb",
    )

    job.execute()

    return job


if __name__ == "__main__":
    main()
