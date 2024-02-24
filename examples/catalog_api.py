"""
This example demonstrates how to use the catalog API.
You can use the python API for fine grained control if configurational specification
does not fit your needs.

You can run this pipeline by: python examples/catalog_api.py
"""

from pathlib import Path

from runnable import Pipeline, Task, get_from_catalog, put_in_catalog


def create_content():
    dir_path = Path("data")
    dir_path.mkdir(parents=True, exist_ok=True)  # Make data folder if it doesn't exist

    with open(dir_path / "hello.txt", "w") as f:
        f.write("Hello from magnus!!")

    # Put the created file in the catalog
    put_in_catalog("data/hello.txt")


def retrieve_content():
    # Get the file from the catalog
    get_from_catalog("data/hello.txt")

    with open("data/hello.txt", "r") as f:
        f.read()


def main():
    # This step creates a file and syncs it to the catalog.
    create = Task(name="create_content", command="examples.catalog_api.create_content")
    # This step retrieves the file from the catalog and prints its content.
    retrieve = Task(
        name="retrieve_content",
        command="examples.catalog_api.retrieve_content",
        terminate_with_success=True,
    )

    create >> retrieve

    pipeline = Pipeline(steps=[create, retrieve], start_at=create, add_terminal_nodes=True)

    # Override the default configuration file with the one that has file-system as the catalog.
    run_log = pipeline.execute(configuration_file="examples/configs/fs-catalog.yaml")
    print(run_log)


if __name__ == "__main__":
    main()
