"""
Example pipeline to demonstrate passing data files between tasks.

You can run this pipeline by:
    python run examples/catalog.py
"""

from runnable import Catalog, Pipeline, ShellTask, Stub


def main():
    # Make the data folder if it does not exist
    set_up = ShellTask(
        name="Setup",
        command="mkdir -p data",
    )

    # create a catalog instruction to put a file into the catalog
    create_catalog = Catalog(put=["data/hello.txt"])
    # This task will create a file in the data folder and attaches the instruction
    # to put the file into the catalog.
    create = ShellTask(
        name="Create Content",
        command='echo "Hello from runnable" >> data/hello.txt',
        catalog=create_catalog,
    )

    # We remove the data folder to ensure that the data folder is cleaned up.
    # This is to show that the retrieve step just does not read from existing data
    # This step is stubbed to prevent any accidental deletion, make it a ShellTask
    first_clean = Stub(
        name="Clean up to get again",
        command="rm -rf data",
    )

    # We create a catalog instruction to retrieve a file from the catalog
    # Here we use "compute_folder_name" to point to the directory of interest.
    # You can alteratively ignore compute_folder_name and get "data/hello.txt"
    # You can use wild card following glob patterns to retrieve multiple files.
    get_catalog = Catalog(get=["data/hello.txt"])
    # This task will retrieve the file from the catalog and attach the instruction
    # to retrieve the file from the catalog before execution.
    retrieve = ShellTask(
        name="Retrieve Content",
        command="cat data/hello.txt",
        catalog=get_catalog,
    )

    # We clean up. Note that this step is stubbed to prevent any accidental deletion,
    # Make it a ShellTask to actually clean up.
    clean_up = Stub(
        name="Clean up",
        command="rm -rf data",
        terminate_with_success=True,
    )

    pipeline = Pipeline(
        steps=[set_up, create, first_clean, retrieve, clean_up],
        add_terminal_nodes=True,
    )

    # override the default configuration to use file-system catalog.
    pipeline.execute(configuration_file="examples/configs/fs-catalog.yaml")

    return pipeline


if __name__ == "__main__":
    main()
