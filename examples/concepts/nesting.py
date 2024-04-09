"""
An example to demonstrate nesting workflows within workflows.


Run this pipeline by:
    python examples/concepts/nesting.py

"""

from typing import List

from runnable import Map, Parallel, Pipeline, PythonTask, Stub


def generate_list():
    return List[int], list(range(2))


def main():
    stub = Stub(name="executable", terminate_with_success=True)
    # A dummy pipeline that does nothing interesting
    stubbed_pipeline = Pipeline(steps=[stub], add_terminal_nodes=True)

    # A map step that executes the stubbed pipeline dynamically
    # This step represents 2 parallel workflows when executed.
    inner_most_map = Map(
        name="inner most",
        branch=stubbed_pipeline,
        iterate_on="array",  # Parameter defined in line #20
        iterate_as="y",
        terminate_with_success=True,
    )

    # A pipeline with map state.
    map_pipeline = Pipeline(steps=[inner_most_map], add_terminal_nodes=True)

    # A parallel step that executes a map_pipeline and stubbed pipeline
    # By nesting a map within the parallel step, the total number of workflows is 4  (2 X 2 = 4)
    nested_parallel = Parallel(
        name="nested parallel",
        branches={"a": map_pipeline, "b": map_pipeline},
        terminate_with_success=True,
    )

    # A pipeline with one nested parallel step
    nested_parallel_pipeline = Pipeline(steps=[nested_parallel], add_terminal_nodes=True)

    list_generator = PythonTask(name="generate list", function=generate_list, returns=["array"])

    # A map step that iterates over array and executes nested_parallel_pipeline
    # The total number of workflows is 50 by this time (2 X 2 X 2 = 8)
    outer_most_map = Map(
        name="outer most",
        branch=nested_parallel_pipeline,
        iterate_on="array",
        iterate_as="x",
        terminate_with_success=True,
    )

    root_pipeline = Pipeline(steps=[list_generator, outer_most_map], add_terminal_nodes=True)

    _ = root_pipeline.execute(configuration_file="examples/configs/fs-catalog-run_log.yaml")

    return root_pipeline


if __name__ == "__main__":
    main()
