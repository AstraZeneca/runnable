from examples.common.functions import read_files, write_files
from runnable import Catalog, Pipeline, PythonTask, ShellTask


def main():
    write_catalog = Catalog(put=["df.csv", "data_folder/data.txt"])
    generate_data = PythonTask(
        name="generate_data_python",
        function=write_files,
        catalog=write_catalog,
    )

    delete_files_command = """
        rm df.csv || true && \
        rm data_folder/data.txt || true
    """
    # delete from local files after generate
    # since its local catalog, we delete to show "get from catalog"
    delete_local_after_generate = ShellTask(
        name="delete_after_generate",
        command=delete_files_command,
    )

    read_catalog = Catalog(get=["df.csv", "data_folder/data.txt"])
    read_data_python = PythonTask(
        name="read_data_python",
        function=read_files,
        catalog=read_catalog,
        terminate_with_success=True,
    )

    pipeline = Pipeline(
        steps=[
            generate_data,
            delete_local_after_generate,
            read_data_python,
        ]
    )
    _ = pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
