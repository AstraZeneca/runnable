"""
A pipeline to demonstrate using the catalog service to create and retrieve content.
Here we use the python API get and put in the catalog.

You can run this pipeline by:
    python run examples/concepts/catalog_api.py
"""

from pathlib import Path

from runnable import get_from_catalog, put_in_catalog


def create_content_in_data_folder():
    """
    Create a data directory and write a file "hello.txt" in the data folder.
    Use the python API put_in_catalog to put the file in the catalog.
    """
    Path("data").mkdir(parents=True, exist_ok=True)
    with open(Path("data") / "hello.txt", "w") as f:
        f.write("Hello from data folder!!")

    put_in_catalog("data/hello.txt")


def create_content_in_another_folder():
    """
    Create a "another" directory and write a file "world.txt" in it.
    Use the python API put_in_catalog to put the file in the catalog.
    """
    Path("another").mkdir(parents=True, exist_ok=True)
    with open(Path("another") / "world.txt", "w") as f:
        f.write("Hello from another folder!!")

    put_in_catalog("another/world.txt")


def retrieve_content_from_both():
    """
    Retrieve the contents of the files from the catalog using the python
    API get_from_catalog.
    Display the contents of the files in data and "another" folder
    """

    get_from_catalog("**/*")

    with open(Path("data") / "hello.txt", "r") as f:
        print(f.read())

    with open(Path("another") / "world.txt", "r") as f:
        print(f.read())


def main():
    from runnable import Pipeline, PythonTask, ShellTask

    # This step creates a file in the data folder and syncs it to the catalog.
    data_create = PythonTask(
        name="create_content_in_data_folder",
        function=create_content_in_data_folder,
    )

    # This step creates a file in the another folder and syncs it to the catalog.
    another_create = PythonTask(
        name="create_content_in_another_folder",
        function=create_content_in_another_folder,
    )

    # Delete the another folder to showcase that the folder will be recreated
    # when we run the retrieve task.
    delete_another_folder = ShellTask(
        name="delete_another_folder",
        command="rm -rf another/",
    )

    # This step retrieves the file from the catalog and prints its content.
    retrieve = PythonTask(
        name="retrieve_content_from_both",
        function=retrieve_content_from_both,
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
