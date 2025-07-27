"""
Runnable pipeline version of multi-CPU PyTorch DDP training.

This demonstrates how to wrap the run_multi_cpu_training function
in a PythonTask and execute it using runnable with parameters from a YAML file.

You can execute this pipeline by:
    python examples/torch/multi_cpu_runnable.py
"""

from examples.torch.multi_cpu import run_multi_cpu_training
from runnable import PythonTask


def main():
    """
    Create and execute a pipeline with the multi-CPU DDP training task using parameters from YAML file.
    """
    # Create a PythonTask that wraps our training function
    training_task = PythonTask(
        name="multi_cpu_training",
        function=run_multi_cpu_training,
        terminate_with_success=True,
    )

    # Convert the task to a pipeline
    pipeline = training_task.as_pipeline()

    # Execute with parameters from the YAML file
    pipeline.execute(parameters_file="examples/torch/multi_cpu_params.yaml")

    return pipeline


if __name__ == "__main__":
    main()
