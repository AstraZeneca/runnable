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
from runnable import TorchJob


def main():
    job = TorchJob(function=hello, num_gpus=2)

    job.execute()

    return job


if __name__ == "__main__":
    main()
