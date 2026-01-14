"""
Example demonstrating a loop that hits max_iterations.

This example shows how the Loop node behaves when max_iterations is reached
before the break condition is met.

You can execute this pipeline by:

    python examples/10_loop/max_iterations_loop.py
"""

import logging

from runnable.sdk import Loop, Pipeline, PythonTask

logger = logging.getLogger(__name__)


def never_stop_task() -> tuple:
    """
    A task that never sets the break condition to True.
    This will cause the loop to hit max_iterations.

    Args:
        counter: The iteration counter from environment variable

    Returns:
        dict: Always returns should_stop=False
    """
    import os

    actual_counter = int(os.environ.get("counter", 0))

    logger.info(f"Executing task iteration #{actual_counter}")

    # This task never wants to stop, so it will hit max_iterations
    return actual_counter, False, f"Completed iteration {actual_counter}"


def check_final_count(counter: int, should_stop: bool) -> tuple:
    """
    Check that we executed the expected number of iterations.

    Args:
        counter: The final iteration counter
        should_stop: The final break condition value

    Returns:
        dict: Validation results
    """
    logger.info(f"Final counter: {counter}, should_stop: {should_stop}")

    # Since we never set should_stop=True, the loop should have hit max_iterations
    # With max_iterations=3, we should see counter=2 (0-indexed: 0, 1, 2)
    expected_final_counter = 2

    assert (
        counter == expected_final_counter
    ), f"Expected counter={expected_final_counter}, got {counter}"
    assert should_stop == False, f"Expected should_stop=False, got {should_stop}"

    return counter, True, True


def never_stop_branch(execute: bool = True):
    """
    Define a branch that never wants to stop.
    """
    task = PythonTask(
        name="never_stop",
        function=never_stop_task,
        returns=["counter", "should_stop", "message"],
    )

    pipeline = Pipeline(steps=[task])

    if execute:
        pipeline.execute()

    return pipeline


def main():
    """
    Main function demonstrating a loop that hits max_iterations.
    """
    print("Running max iterations loop example...")

    # Create a loop with a low max_iterations that will be reached
    max_iter_loop = Loop(  # [concept:loop]
        name="max_iter_loop",
        branch=never_stop_branch(execute=False),  # [concept:branch-pipeline]
        max_iterations=3,  # [concept:max-iterations] Low limit - will be reached
        break_on="should_stop",  # [concept:break-condition] Never becomes True
        index_as="counter",  # [concept:index-variable] Available as environment variable
    )

    # After the loop, validate that we hit max_iterations
    validate_task = PythonTask(
        name="validate_max_iter",
        function=check_final_count,
        returns=[
            "final_counter",
            "hit_max_iterations",
            "validation_passed",
        ],
        terminate_with_success=True,
    )

    # Create the main pipeline
    pipeline = Pipeline(
        name="max_iterations_example", steps=[max_iter_loop, validate_task]
    )  # [concept:pipeline]

    # Execute the pipeline
    # The loop will run exactly max_iterations (3) times: iterations 0, 1, 2
    pipeline.execute()  # [concept:execution]

    return pipeline


if __name__ == "__main__":
    main()
