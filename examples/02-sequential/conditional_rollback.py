"""
Demonstrate parameter rollback from conditional branches.

When a conditional branch executes and succeeds, parameters set within
that branch roll back to the parent scope.

Execute with:
    python examples/10-branch-parameters/conditional_rollback.py
"""

from examples.common.functions import (
    set_conditional_heads_param,
    set_conditional_tails_param,
    verify_conditional_rollback,
)
from runnable import Conditional, Pipeline, PythonTask


def decide_heads():
    """Return 'heads' to select heads branch."""
    return "heads"


def main():
    # Create branch pipelines that set parameters
    heads_pipeline = PythonTask(
        name="heads_task",
        function=set_conditional_heads_param,
        returns=["branch_param"],
    ).as_pipeline()

    tails_pipeline = PythonTask(
        name="tails_task",
        function=set_conditional_tails_param,
        returns=["branch_param"],
    ).as_pipeline()

    # Conditional node selects branch based on 'choice' parameter
    conditional = Conditional(
        name="conditional",
        branches={"heads": heads_pipeline, "tails": tails_pipeline},
        parameter="choice",
    )

    # Task to set the choice parameter
    decide_task = PythonTask(
        name="decide",
        function=decide_heads,
        returns=["choice"],
    )

    # Task to verify the parameter rolled back from branch
    verify_task = PythonTask(
        name="verify",
        function=verify_conditional_rollback,
        terminate_with_success=True,
    )

    pipeline = Pipeline(steps=[decide_task, conditional, verify_task])
    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
