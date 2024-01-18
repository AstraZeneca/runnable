from typing import List

from pydantic import create_model

from magnus import Map, Parallel, Pipeline, Stub, Task


def generate_list():
    return create_model(
        "DynamicModel",
        array=(List[int], list(range(2))),
    )()


def main():
    stub = Stub(name="executable", terminate_with_success=True)
    # A dummy pipeline that does nothing interesting
    stubbed_pipeline = Pipeline(steps=[stub], start_at=stub, add_terminal_nodes=True)

    # A map step that executes the stubbed pipeline dynamically
    # This step represents 2 parallel workflows when executed.
    inner_most_map = Map(
        name="inner most",
        branch=stubbed_pipeline,
        iterate_on="array",
        iterate_as="y",
        terminate_with_success=True,
    )

    # A pipeline with map state.
    map_pipeline = Pipeline(steps=[inner_most_map], start_at=inner_most_map, add_terminal_nodes=True)

    # A parallel step that executes a map_pipeline and stubbed pipeline
    # By nesting a map within the parallel step, the total number of workflows is 4  (2 X 2 = 4)
    nested_parallel = Parallel(
        name="nested parallel",
        branches={"a": map_pipeline, "b": map_pipeline},
        terminate_with_success=True,
    )

    # A pipeline with one nested parallel step
    nested_parallel_pipeline = Pipeline(steps=[nested_parallel], start_at=nested_parallel, add_terminal_nodes=True)

    list_generator = Task(name="generate list", command="examples.concepts.nesting.generate_list")

    # A map step that iterates over array and executes nested_parallel_pipeline
    # The total number of workflows is 50 by this time (2 X 2 X 2 = 8)
    outer_most_map = Map(
        name="outer most",
        branch=nested_parallel_pipeline,
        iterate_on="array",
        iterate_as="x",
        terminate_with_success=True,
    )

    list_generator >> outer_most_map

    root_pipeline = Pipeline(steps=[list_generator, outer_most_map], start_at=list_generator, add_terminal_nodes=True)

    _ = root_pipeline.execute(configuration_file="examples/configs/fs-catalog-run_log.yaml")


if __name__ == "__main__":
    main()
