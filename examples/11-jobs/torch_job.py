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

from runnable import TorchJob


def main():
    job = TorchJob(
        args_to_torchrun={"nproc_per_node": "2", "backend": "gloo"},
        script_to_call="examples/common/script.py",
    )

    job.execute(parameters_file="examples/common/initial_parameters.yaml")

    return job


if __name__ == "__main__":
    main()
