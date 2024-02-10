"""
A pipeline to demonstrate using the catalog service to create and retrieve content.

You can run this pipeline by:
    python run examples/concepts/catalog.py
"""

from pathlib import Path


def create_content_in_data_folder():
    """
    Create a data directory and write a file "hello.txt" in the data folder.
    """
    Path("data").mkdir(parents=True, exist_ok=True)
    with open(Path("data") / "hello.txt", "w") as f:
        f.write("Hello from data folder!!")


def create_content_in_another_folder():
    """
    Create a "another" directory and write a file "world.txt" in it.
    """
    Path("another").mkdir(parents=True, exist_ok=True)
    with open(Path("another") / "world.txt", "w") as f:
        f.write("Hello from another folder!!")


def retrieve_content_from_both():
    """
    Display the contents of the files in data and "another" folder
    """
    with open(Path("data") / "hello.txt", "r") as f:
        print(f.read())

    with open(Path("another") / "world.txt", "r") as f:
        print(f.read())


def main():
    from magnus import Catalog, Pipeline, Task

    # This step creates a file in the data folder and syncs it to the catalog.
    data_catalog = Catalog(put=["data/hello.txt"])
    data_create = Task(
        name="create_content_in_data_folder",
        command="examples.concepts.catalog.create_content_in_data_folder",
        catalog=data_catalog,
    )

    # This step creates a file in the another folder and syncs it to the catalog.
    another_catalog = Catalog(put=["another/world.txt"])
    another_create = Task(
        name="create_content_in_another_folder",
        command="examples.concepts.catalog.create_content_in_another_folder",
        catalog=another_catalog,
    )

    # Delete the another folder to showcase that the folder will be recreated
    # when we run the retrieve task.
    delete_another_folder = Task(
        name="delete_another_folder",
        command="rm -rf another/",
        command_type="shell",
    )

    # This step retrieves the file from the catalog and prints its content.
    all_catalog = Catalog(get=["**/*"])
    retrieve = Task(
        name="retrieve_content_from_both",
        command="examples.concepts.catalog.retrieve_content_from_both",
        catalog=all_catalog,
        terminate_with_success=True,
    )

    data_create >> another_create >> delete_another_folder >> retrieve

    pipeline = Pipeline(
        steps=[data_create, another_create, retrieve, delete_another_folder],
        start_at=data_create,
        add_terminal_nodes=True,
    )

    # Override the default configuration file with the one that has file-system as the catalog.
    _ = pipeline.execute(configuration_file="examples/configs/fs-catalog.yaml")


if __name__ == "__main__":
    main()
