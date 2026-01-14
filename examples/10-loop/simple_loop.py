"""
Simple loop example using the Loop node.

This example demonstrates a basic loop that counts up to a limit.

You can execute this pipeline by:

    python examples/10_loop/simple_loop.py
"""

import logging
import os

from runnable.sdk import Loop, Pipeline, PythonTask

logger = logging.getLogger(__name__)


def counter_task():
    """
    A simple counter task that increments until reaching 3.

    Returns:
        bool: True when counter reaches 3, False otherwise
    """
    # Get the iteration counter from environment variable set by loop node
    counter = int(os.environ.get("counter", 0))

    logger.info(f"Counter task executing, counter = {counter}")

    # Stop when we reach 3
    should_stop = counter >= 3

    logger.info(f"Should stop: {should_stop}")

    return should_stop


def counter_branch(execute: bool = True):
    """
    Define the counter branch pipeline.
    """
    task = PythonTask(
        name="count",
        function=counter_task,
        returns=["should_stop"],  # Single return value
    )

    pipeline = Pipeline(steps=[task])

    if execute:
        pipeline.execute()

    return pipeline


def main():
    """
    Main function demonstrating a simple loop.
    """
    print("Running simple loop example...")

    # Create the loop
    counter_loop = Loop(
        name="counter_loop",
        branch=counter_branch(execute=False),
        max_iterations=5,  # Safety limit
        break_on="should_stop",  # Exit when should_stop becomes True
        index_as="counter",  # Environment variable name
    )

    # Create the main pipeline
    pipeline = Pipeline(name="simple_loop_example", steps=[counter_loop])

    # Execute the pipeline
    pipeline.execute()

    print("✅ Simple loop example completed successfully!")
    return pipeline


if __name__ == "__main__":
    main()
