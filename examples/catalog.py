from magnus import Catalog, Pipeline, Stub, Task


def main():
    # Make the data folder if it does not exist
    set_up = Task(name="Setup", command="mkdir -p data", command_type="shell")

    # create a catalog instruction to put a file into the catalog
    create_catalog = Catalog(put=["data/hello.txt"])
    # This task will create a file in the data folder and attaches the instruction
    # to put the file into the catalog.
    create = Task(
        name="Create Content",
        command='echo "Hello from magnus" >> data/hello.txt',
        command_type="shell",
        catalog=create_catalog,
    )

    # We remove the data folder to ensure that the data folder is cleaned up.
    # This is to show that the retrieve step just does not read from existing data
    # This step is stubbed to prevent any accidental deletion, make it a Task
    first_clean = Stub(
        name="Clean up to get again",
        command="rm -rf data",
        command_type="shell",
    )

    # We create a catalog instruction to retrieve a file from the catalog
    # Here we use "compute_folder_name" to point to the directory of interest.
    # You can alteratively ignore compute_folder_name and get "data/hello.txt"
    # You can use wild card following glob patterns to retrieve multiple files.
    get_catalog = Catalog(compute_data_folder="data", get=["hello.txt"])
    # This task will retrieve the file from the catalog and attach the instruction
    # to retrieve the file from the catalog before execution.
    retrieve = Task(
        name="Retrieve Content",
        command="cat data/hello.txt",
        command_type="shell",
        catalog=get_catalog,
    )

    # We clean up. Note that this step is stubbed to prevent any accidental deletion,
    # Make it a Task to actually clean up.
    clean_up = Stub(
        name="Clean up",
        command="rm -rf data",
        command_type="shell",
        terminate_with_success=True,
    )

    # link all the steps of the pipeline
    set_up >> create >> first_clean >> retrieve >> clean_up

    pipeline = Pipeline(
        steps=[set_up, create, first_clean, retrieve, clean_up],
        start_at=set_up,
        add_terminal_nodes=True,
    )

    # override the default configuration to use file-system catalog.
    pipeline.execute(configuration_file="examples/configs/fs-catalog.yaml")


if __name__ == "__main__":
    main()
