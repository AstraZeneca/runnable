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

# Outside of loop, processed is a list of all the processed chunks.
# This is also called as the reduce pattern.
assert processed == [chunk * 10 for chunk in chunks]

Run this pipeline as:
    python examples/07-map/map.py
"""

from examples.common.functions import (
    assert_default_reducer,
    process_chunk,
    read_processed_chunk,
)
from runnable import Map, NotebookTask, Pipeline, PythonTask, ShellTask


def iterable_branch(execute: bool = True):
    """
    Use the pattern of using "execute" to control the execution of the pipeline.

    The same pipeline can be run independently from the command line.

    WARNING: If the execution is not controlled by "execute", the pipeline will be executed
    even during the definition of the branch in parallel steps.
    """
    # The python function to process a single chunk of data.
    # In the example, we are multiplying the chunk by 10.
    process_chunk_task_python = PythonTask(
        name="execute_python",
        function=process_chunk,
        returns=["processed_python"],
    )

    # return parameters within a map branch have to be unique
    # The notebook takes in the value of processed_python as an argument.
    # and returns a new parameter "processed_notebook" which is 10*processed_python
    process_chunk_task_notebook = NotebookTask(
        name="execute_notebook",
        notebook="examples/common/process_chunk.ipynb",
        returns=["processed_notebook"],
    )

    # following the pattern, the shell takes in the value of processed_notebook as an argument.
    # and returns a new parameter "processed_shell" which is 10*processed_notebook.
    shell_command = """
    if [ "$processed_python" = $( expr 10 '*' "$chunk" ) ] \
        && [ "$processed_notebook" = $( expr 10 '*' "$processed_python" ) ] ; then
            echo "yaay"
        else
            echo "naay"
            exit 1;
    fi
    export processed_shell=$( expr 10 '*' "$processed_notebook")
    """

    process_chunk_task_shell = ShellTask(
        name="execute_shell",
        command=shell_command,
        returns=["processed_shell"],
    )

    # A downstream step of process_<python, notebook, shell> which reads the parameter "processed".
    # The value of processed is within the context of the branch.
    # For example, for chunk=1, the value of processed_python is chunk*10 = 10
    # the value of processed_notebook is processed_python*10 = 100
    # the value of processed_shell is processed_notebook*10 = 1000
    read_chunk = PythonTask(
        name="read processed chunk",
        function=read_processed_chunk,
        terminate_with_success=True,
    )

    pipeline = Pipeline(
        steps=[
            process_chunk_task_python,
            process_chunk_task_notebook,
            process_chunk_task_shell,
            read_chunk,
        ],
    )

    if execute:
        pipeline.execute()

    return pipeline


def main():
    # Create a map state which iterates over a list of chunks.
    # chunk is the value of the iterable.
    map_state = Map(
        name="map state",
        iterate_on="chunks",
        iterate_as="chunk",
        branch=iterable_branch(execute=False),
    )

    # Outside of the loop, processed is a list of all the processed chunks.
    # This is also called as the reduce pattern.
    # the value of processed_python is [10, 20, 30]
    # the value of processed_notebook is [100, 200, 300]
    # the value of processed_shell is [1000, 2000, 3000]
    collect = PythonTask(
        name="collect",
        function=assert_default_reducer,
        terminate_with_success=True,
    )

    pipeline = Pipeline(steps=[map_state, collect])

    pipeline.execute(parameters_file="examples/common/initial_parameters.yaml")

    return pipeline


if __name__ == "__main__":
    main()
