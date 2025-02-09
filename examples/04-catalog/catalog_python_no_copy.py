from examples.common.functions import read_files, write_files
from runnable import Catalog, Pipeline, PythonTask


def main():
    write_catalog = Catalog(put=["df.csv", "data_folder/data.txt"])
    generate_data = PythonTask(
        name="generate_data_python",
        function=write_files,
        catalog=write_catalog,
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
            read_data_python,
        ]
    )
    _ = pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
