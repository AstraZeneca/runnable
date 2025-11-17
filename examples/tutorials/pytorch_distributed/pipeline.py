"""
PyTorch Distributed Training Pipeline

This tutorial demonstrates single-node distributed training using PyTorch's
DistributedDataParallel (DDP) on CPU cores with Runnable orchestration.

Key Features Demonstrated:
- Multi-process distributed training on CPU
- PyTorch DistributedDataParallel (DDP)
- Gradient synchronization across processes
- Model checkpoint saving and loading
- Training metrics aggregation
- Runnable pipeline orchestration

Pipeline Flow:
Prepare Data → Run Distributed Training → Evaluate Model → Generate Report

The distributed training internally spawns multiple processes:
[Process 0 | Process 1 | Process 2 | Process 3]
    \          |          |          /
     \         |          |         /
      Gradient Synchronization (DDP)

Run this pipeline with:
    uv run --group tutorial examples/tutorials/pytorch_distributed/pipeline.py

The pipeline will:
1. Prepare synthetic dataset for distributed training
2. Launch multi-process distributed training using PyTorch DDP
3. Aggregate training metrics from all processes
4. Evaluate the final trained model
5. Generate comprehensive training report
"""

from examples.tutorials.pytorch_distributed.functions import (
    evaluate_distributed_model,
    prepare_training_data,
    run_distributed_training,
)
from runnable import Catalog, Pipeline, PythonTask, metric, pickled


def main():
    """
    Create and execute the PyTorch distributed training pipeline.
    """

    # Task 1: Prepare training data
    prepare_data_task = PythonTask(
        function=prepare_training_data,
        name="prepare_data",
        returns=[pickled("dataset_info")],
    )

    # Task 2: Run distributed training across multiple processes
    distributed_training_task = PythonTask(
        function=run_distributed_training,
        name="distributed_training",
        returns=[
            pickled("training_results"),
            metric("avg_final_accuracy"),  # Track final accuracy as metric
            metric("total_training_time"),  # Track training time as metric
        ],
        catalog=Catalog(
            put=[
                "distributed_model_checkpoint.pt",  # Final checkpoint
                "checkpoint_latest.pt",             # Latest checkpoint for resuming
                "checkpoint_epoch_*.pt",            # All epoch checkpoints
                "distributed_training_results.json",
                "process_*_stdout.txt",             # Process stdout captures
                "process_*_stderr.txt",             # Process stderr captures
                "process_*_combined.txt",           # Process combined output
                "process_*_output.log"              # Process-specific log files
            ]
        ),
    )

    # Task 3: Evaluate the trained model
    evaluate_model_task = PythonTask(
        function=evaluate_distributed_model,
        name="evaluate_model",
        returns=[
            pickled("evaluation_results"),
            metric("test_accuracy"),  # Track test accuracy as metric
        ],
        terminate_with_success=True,
    )

    # Create the pipeline
    pipeline = Pipeline(
        steps=[
            prepare_data_task,
            distributed_training_task,
            evaluate_model_task,
        ]
    )

    # Execute the pipeline with parameters from YAML file
    pipeline.execute(
        parameters_file="examples/tutorials/pytorch_distributed/parameters.yaml"
    )

    return pipeline


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    import multiprocessing as mp

    # Use 'spawn' method for better compatibility with distributed training
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    main()
