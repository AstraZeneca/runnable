from typing import List

from pydantic import create_model


def sequence_generator():
    return create_model("DynamicModel", sequence=(List[int], [1, 2, 3, 4, 5]))()


def execute_on_every_element(i: int):
    print(i)


def main():
    from magnus import Map, Pipeline, Task

    execute = Task(
        name="execute",
        command="examples.concepts.map.execute_on_every_element",
        terminate_with_success=True,
    )

    execute_branch = Pipeline(steps=[execute], start_at=execute, add_terminal_nodes=True)

    generate = Task(name="generate sequence", command="examples.concepts.map.sequence_generator")
    iterate_and_execute = Map(
        name="iterate and execute",
        branch=execute_branch,
        iterate_on="sequence",
        iterate_as="i",
        terminate_with_success=True,
    )

    generate >> iterate_and_execute

    pipeline = Pipeline(steps=[generate, iterate_and_execute], start_at=generate, add_terminal_nodes=True)

    _ = pipeline.execute(configuration_file="examples/configs/fs-catalog-run_log.yaml")


if __name__ == "__main__":
    main()
