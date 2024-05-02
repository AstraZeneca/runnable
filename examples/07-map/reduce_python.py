"""
map states allows to repeat a branch for each value of an iterable.

The below example can written, in python, as:

chunks = [1, 2, 3]

for chunk in chunks:
    # Any task within the pipeline can access the value of chunk as an argument.
    processed = process_chunk(chunk)

    # The value of processed for every iteration is the value returned by the steps
    # of the current execution. For example, the value of processed
    # for chunk=1, is chunk*10 = 10 for downstream steps.
    read_processed_chunk(chunk, processed)

It is possible to use a custom reducer, for example, this reducer is a max of the collection.
# Once the reducer is applied, processed is reduced to a single value.
assert processed == max(chunk * 10 for chunk in chunks)
"""

from examples.common.functions import (
    assert_custom_reducer,
    process_chunk,
    read_processed_chunk,
)
from runnable import Map, Pipeline, PythonTask


def iterable_branch(execute: bool = True):
    """
    Use the pattern of using "execute" to control the execution of the pipeline.

    The same pipeline can be run independently from the command line.

    WARNING: If the execution is not controlled by "execute", the pipeline will be executed
    even during the definition of the branch in parallel steps.
    """
    # The python function to process a single chunk of data.
    # In the example, we are multiplying the chunk by 10.
    process_chunk_task = PythonTask(
        name="execute",
        function=process_chunk,
        returns=["processed"],
    )

    # A downstream step of process_chunk which reads the parameter "processed".
    # The value of processed is within the context of the branch.
    # For example, for the value of chunk = 1, processed will be 10.
    # read_processed_chunk will receive the value of 10.
    read_chunk = PythonTask(
        name="read processed chunk",
        function=read_processed_chunk,
        terminate_with_success=True,
    )

    pipeline = Pipeline(
        steps=[process_chunk_task, read_chunk],
        add_terminal_nodes=True,
    )

    if execute:
        pipeline.execute()

    return pipeline


def main():
    # Create a map state which iterates over a list of chunks.
    # chunk is the value of the iterable.
    # Upon completion of the map state, all the parameters of the tasks
    # within the pipeline will be processed by the reducer.
    # In this case, the reducer is the max of all the processed chunks.
    map_state = Map(
        name="map state",
        iterate_on="chunks",
        iterate_as="chunk",
        reducer="lambda *x: max(x)",
        branch=iterable_branch(execute=False),
    )

    collect = PythonTask(
        name="collect",
        function=assert_custom_reducer,
        terminate_with_success=True,
    )

    pipeline = Pipeline(steps=[map_state, collect])

    pipeline.execute(parameters_file="examples/common/initial_parameters.yaml")

    return pipeline


if __name__ == "__main__":
    main()
