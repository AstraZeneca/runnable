"""
Runnable job version of multi-CPU PyTorch DDP training.

This demonstrates how to wrap the run_multi_cpu_training function
in a PythonJob and execute it using runnable with parameters from a YAML file.

You can execute this job by:
    python examples/torch/multi_cpu_job.py
"""

from examples.torch.multi_cpu import run_multi_cpu_training
from runnable import PythonJob


def main():
    """
    Create and execute a job with the multi-CPU DDP training function using parameters from YAML file.
    """
    # Create a PythonJob that wraps our training function
    training_job = PythonJob(
        function=run_multi_cpu_training,
    )

    # Execute with parameters from the YAML file
    training_job.execute(parameters_file="examples/torch/multi_cpu_params.yaml")

    return training_job


if __name__ == "__main__":
    main()
