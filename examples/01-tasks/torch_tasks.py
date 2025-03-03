"""
You can execute this pipeline by:

    python examples/01-tasks/python_tasks.py

The stdout of "Hello World!" would be captured as execution
log and stored in the catalog.

An example of the catalog structure:

.catalog
└── baked-heyrovsky-0602
    └── hello.execution.log

2 directories, 1 file


The hello.execution.log has the captured stdout of "Hello World!".
"""

from examples.common.functions import hello
from runnable import Pipeline, Torch


def main():
    torch_task = Torch(
        function=hello,
        name="torch",
        terminate_with_success=True,
        max_restarts=2,
        overrides={"argo": "cpu-machine"},
    )

    # The pipeline has only one step.
    pipeline = Pipeline(steps=[torch_task])

    pipeline.execute()
    return pipeline


if __name__ == "__main__":
    main()
