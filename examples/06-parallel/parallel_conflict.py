"""
Demonstrate parameter conflict resolution in parallel branches.

When multiple branches set the same parameter, last write wins
based on dictionary iteration order.

Execute with:
    python examples/10-branch-parameters/parallel_conflict.py
"""

from examples.common.functions import set_shared_param_a, set_shared_param_b
from runnable import Parallel, Pipeline, PythonTask


def main():
    # Both branches set the same parameter name
    branch1_pipeline = PythonTask(
        name="branch1_task",
        function=set_shared_param_a,
        returns=["shared"],
    ).as_pipeline()

    branch2_pipeline = PythonTask(
        name="branch2_task",
        function=set_shared_param_b,
        returns=["shared"],
    ).as_pipeline()

    parallel = Parallel(
        name="parallel",
        branches={"branch1": branch1_pipeline, "branch2": branch2_pipeline},
    )

    pipeline = Pipeline(steps=[parallel])
    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
