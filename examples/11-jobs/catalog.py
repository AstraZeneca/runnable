from examples.common.functions import write_files
from runnable import Catalog, PythonJob


def main():
    write_catalog = Catalog(put=["df.csv", "data_folder/data.txt"])
    job = PythonJob(
        function=write_files,
        catalog=write_catalog,
    )

    job.execute()

    return job


if __name__ == "__main__":
    main()
