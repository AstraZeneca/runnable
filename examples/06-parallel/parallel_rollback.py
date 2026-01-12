"""
Demonstrate parameter rollback from parallel branches.

When parallel branches execute and succeed, parameters from all branches
roll back to the parent scope.

Execute with:
    python examples/10-branch-parameters/parallel_rollback.py
"""

from examples.common.functions import (
    set_parallel_branch1,
    set_parallel_branch2,
    set_parallel_branch3,
    verify_parallel_rollback,
)
from runnable import Parallel, Pipeline, PythonTask


def main():
    # Create branch pipelines that set different parameters
    branch1_pipeline = PythonTask(
        name="branch1_task",
        function=set_parallel_branch1,
        returns=["result1"],
    ).as_pipeline()

    branch2_pipeline = PythonTask(
        name="branch2_task",
        function=set_parallel_branch2,
        returns=["result2"],
    ).as_pipeline()

    branch3_pipeline = PythonTask(
        name="branch3_task",
        function=set_parallel_branch3,
        returns=["result3"],
    ).as_pipeline()

    # Parallel node executes all branches
    parallel = Parallel(
        name="parallel",
        branches={
            "branch1": branch1_pipeline,
            "branch2": branch2_pipeline,
            "branch3": branch3_pipeline,
        },
    )

    # Task to verify all parameters rolled back from branches
    verify_task = PythonTask(
        name="verify",
        function=verify_parallel_rollback,
        terminate_with_success=True,
    )

    pipeline = Pipeline(steps=[parallel, verify_task])
    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
