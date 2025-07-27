"""
Runnable version of single CPU PyTorch training.

This demonstrates how to wrap the run_single_cpu_training function
in a PythonTask and execute it using runnable with parameters from a YAML file.

You can execute this pipeline by:
    python examples/torch/single_cpu_runnable.py
"""

from examples.torch.single_cpu import run_single_cpu_training
from runnable import PythonTask


def main():
    """
    Create and execute a pipeline with the single CPU training task using parameters from YAML file.
    """
    # Create a PythonTask that wraps our training function
    training_task = PythonTask(
        name="single_cpu_training",
        function=run_single_cpu_training,
        terminate_with_success=True,
    )

    # Convert the task to a pipeline
    pipeline = training_task.as_pipeline()

    # Execute with parameters from the YAML file
    pipeline.execute(parameters_file="examples/torch/single_cpu_params.yaml")

    return pipeline


if __name__ == "__main__":
    main()
