from magnus import Catalog, Pipeline, Task


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
        terminate_with_success=True,
    )

    set_up >> create

    pipeline = Pipeline(
        steps=[set_up, create],
        start_at=set_up,
        add_terminal_nodes=True,
    )

    # override the default configuration to use file-system catalog.
    pipeline.execute(configuration_file="examples/configs/fs-catalog.yaml")


if __name__ == "__main__":
    main()
