"""
Dynamic Map Example - iterate_on from a previous step's return value.

This example shows how to dynamically generate the list of items to iterate over
at runtime, rather than defining them statically in a parameters file.

You can execute this pipeline by:

    uv run examples/07-map/dynamic_map.py
"""

from runnable import Map, Pipeline, PythonTask


def generate_chunks() -> list[int]:
    """
    Dynamically generate the list of items to iterate over.

    In real scenarios, this could:
    - Query a database for records to process
    - List files in a directory
    - Fetch work items from an API
    - Generate based on runtime conditions
    """
    print("Generating chunks dynamically...")
    return [1, 2, 3, 4, 5]


def process_chunk(chunk: int) -> int:
    """Process a single chunk - runs once per item in the list."""
    result = chunk * 10
    print(f"Processing chunk {chunk} -> {result}")
    return result


def collect_results(processed: list[int]):
    """Collect all processed results after the map completes."""
    print(f"All results: {processed}")
    print(f"Sum: {sum(processed)}")


def iterable_branch(execute: bool = True):
    """The branch pipeline that runs for each item."""
    process_task = PythonTask(
        name="process",
        function=process_chunk,
        returns=["processed"],
        terminate_with_success=True,
    )

    pipeline = Pipeline(steps=[process_task])

    if execute:
        pipeline.execute()

    return pipeline


def main():
    # Step 1: Generate the list of items to iterate over
    generate_task = PythonTask(
        name="generate_chunks",
        function=generate_chunks,
        returns=["chunks"],  # This becomes available as a parameter
    )

    # Step 2: Map over the dynamically generated list
    map_state = Map(
        name="process_all",
        iterate_on="chunks",  # References the return value from generate_chunks
        iterate_as="chunk",  # Variable name available in each iteration
        branch=iterable_branch(execute=False),
    )

    # Step 3: Collect results after all iterations complete
    collect_task = PythonTask(
        name="collect",
        function=collect_results,
    )

    pipeline = Pipeline(steps=[generate_task, map_state, collect_task])
    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
