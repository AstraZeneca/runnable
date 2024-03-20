"""
An example pipeline of using "map" to process a sequence of nodes repeatedly over a iterable
parameter.

The start_index argument for the function process_chunk is dynamically set by iterating over chunks.

If the argument start_index is not provided, you can still access the current value by
runnable_MAP_VARIABLE environment variable. The environment variable runnable_MAP_VARIABLE
is a dictionary with keys as iterate_as.

Run this pipeline by:
    python examples/concepts/map.py
"""


def chunk_files():
    """
    Identify the number of chunks and files to execute per batch.

    Set the parameter "chunks" to be the start indexes of batch.
    Set the parameter "stride" to be the number of files to
    execute per batch.
    """
    return 10, list(range(0, 50, 10))
    # create_model(
    #     "DynamicModel",
    #     chunks=(List[int], list(range(0, 50, 10))),
    #     stride=(int, 10),
    # )()


def process_chunk(stride: int, start_index: int):
    """
    The function processes a chunk of files.
    The files between the start_index and the start_index + stride
    are processed per chunk.
    """
    print("stride", stride, type(stride))
    print("start_index", start_index, type(start_index))
    for i in range(start_index, start_index + stride, stride):
        pass


def main():
    """
    The pythonic equivalent of the following pipeline.

    chunks = chunk_files()

    for start_index in chunks.chunks:
        process_chunk(chunks.stride, start_index)

    """
    from runnable import Map, Pipeline, PythonTask

    execute = PythonTask(
        name="execute",
        function=process_chunk,
        terminate_with_success=True,
    )

    execute_branch = Pipeline(steps=[execute], start_at=execute, add_terminal_nodes=True)

    generate = PythonTask(
        name="chunk files",
        function=chunk_files,
        returns=["stride", "chunks"],
    )
    iterate_and_execute = Map(
        name="iterate and execute",
        branch=execute_branch,
        iterate_on="chunks",  # iterate on chunks parameter set by execute step
        iterate_as="start_index",  # expose the current start_index as the iterate_as parameter
        terminate_with_success=True,
    )

    generate >> iterate_and_execute

    pipeline = Pipeline(steps=[generate, iterate_and_execute], start_at=generate, add_terminal_nodes=True)

    _ = pipeline.execute(configuration_file="examples/configs/fs-catalog-run_log.yaml")


if __name__ == "__main__":
    main()
