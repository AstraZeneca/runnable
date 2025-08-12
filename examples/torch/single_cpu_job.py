"""
Runnable job version of single CPU PyTorch training.

This demonstrates how to wrap the run_single_cpu_training function
in a PythonJob and execute it using runnable with parameters from a YAML file.

You can execute this job by:
    python examples/torch/single_cpu_job.py
"""

from examples.torch.single_cpu import run_single_cpu_training
from runnable import PythonJob


def main():
    """
    Create and execute a job with the single CPU training function using parameters from YAML file.
    """
    # Create a PythonJob that wraps our training function
    training_job = PythonJob(
        function=run_single_cpu_training,
    )

    # Execute with parameters from the YAML file
    training_job.execute(parameters_file="examples/torch/single_cpu_params.yaml")

    return training_job


if __name__ == "__main__":
    main()
