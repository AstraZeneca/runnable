import os

from examples.standard_pytorch.train_distributed import train_distributed as train_main

from runnable import Catalog, PythonJob


def main():
    job = PythonJob(
        function=train_main,
        catalog=Catalog(
            put=[
                "training_output/*",
            ]
        ),
    )

    job.execute()

    return job


if __name__ == "__main__":
    os.environ["RUNNABLE_PARAMETERS_FILE"] = (
        "examples/standard_pytorch/train_parameters.yaml"
    )
    main()
