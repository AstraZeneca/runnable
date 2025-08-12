"""
You can execute this pipeline by:

    python examples/04-catalog/catalog_python.py
"""

from examples.common.functions import check_files_do_not_exist, write_files
from runnable import Catalog, Pipeline, PythonTask, ShellTask


def main():
    write_catalog = Catalog(put=["df.csv", "data_folder/data.txt"], store_copy=False)
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

    # Since store_copy was set to False, this step should fail
    check_files_do_not_exist_task = PythonTask(
        name="check_files_do_not_exist",
        function=check_files_do_not_exist,
        terminate_with_success=True,
    )

    pipeline = Pipeline(
        steps=[
            generate_data,
            delete_local_after_generate,
            check_files_do_not_exist_task,
        ]
    )
    _ = pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
