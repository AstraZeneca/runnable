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

.catalog
└── silly-joliot-0610
    ├── data_folder
    │   └── data.txt
    ├── deleteaftergenerate.execution.log
    ├── deleteaftergeneratenotebook.execution.log
    ├── deleteaftergeneratepython.execution.log
    ├── deleteaftergenerateshell.execution.log
    ├── df.csv
    ├── examples
    │   └── common
    │       └── read_files_out.ipynb
    ├── generatedata.execution.log
    ├── readdatanotebook.execution.log
    ├── readdatapy.execution.log
    └── readdatashell.execution.log

5 directories, 11 files

Run this pipeline as:
    python examples/04-catalog/catalog.py

"""

from examples.common.functions import read_files, write_files
from runnable import Catalog, NotebookTask, Pipeline, PythonTask, ShellTask


def main():
    write_catalog = Catalog(put=["df.csv", "data_folder/data.txt"])
    generate_data = PythonTask(
        name="generate_data",
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
        name="read_data_py",
        function=read_files,
        catalog=read_catalog,
    )

    delete_local_after_python_get = ShellTask(
        name="delete_after_generate_python",
        command=delete_files_command,
    )

    read_data_shell_command = """
    (ls df.csv >> /dev/null 2>&1 && echo yes) || exit 1 && \
    (ls data_folder/data.txt >> /dev/null 2>&1 && echo yes) || exit 1
    """
    read_data_shell = ShellTask(
        name="read_data_shell",
        command=read_data_shell_command,
        catalog=read_catalog,
    )

    delete_local_after_shell_get = ShellTask(
        name="delete_after_generate_shell",
        command=delete_files_command,
    )

    read_data_notebook = NotebookTask(
        notebook="examples/common/read_files.ipynb",
        name="read_data_notebook",
        catalog=read_catalog,
    )

    delete_local_after_notebook_get = ShellTask(
        name="delete_after_generate_notebook",
        command=delete_files_command,
        terminate_with_success=True,
    )

    pipeline = Pipeline(
        steps=[
            generate_data,
            delete_local_after_generate,
            read_data_python,
            delete_local_after_python_get,
            read_data_shell,
            delete_local_after_shell_get,
            read_data_notebook,
            delete_local_after_notebook_get,
        ]
    )
    _ = pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
