"""
Demonstrates moving files within tasks.

- generate_data: creates df.csv and data_folder/data.txt

- delete_local_after_generate: deletes df.csv and data_folder/data.txt
    This step ensures that the local files are deleted after the step

- read_data_py: reads df.csv and data_folder/data.txt

- delete_local_after_python_get: deletes df.csv and data_folder/data.txt
    This step ensures that the local files are deleted after the step

- read_data_shell: reads df.csv and data_folder/data.txt

- delete_local_after_shell_get: deletes df.csv and data_folder/data.txt
    This step ensures that the local files are deleted after the step

- read_data_notebook: reads df.csv and data_folder/data.txt

- delete_local_after_notebook_get: deletes df.csv and data_folder/data.txt

Use this pattern to move files that are not dill friendly.

All the files are stored in catalog.


Run this pipeline as:
    python examples/04-catalog/catalog.py

"""

from runnable import Catalog, Pipeline, ShellTask


def main():
    write_catalog = Catalog(put=["df.csv", "data_folder/data.txt"])
    fail_immediately = ShellTask(
        name="fail_immediately",
        command="exit 1",
        catalog=write_catalog,
        terminate_with_failure=True,
    )

    success_node = ShellTask(
        name="success_node",
        command="echo 'success'",
        terminate_with_success=True,
    )
    success_pipeline = Pipeline(steps=[success_node])

    fail_immediately.on_failure = success_pipeline

    pipeline = Pipeline(steps=[fail_immediately])
    _ = pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
