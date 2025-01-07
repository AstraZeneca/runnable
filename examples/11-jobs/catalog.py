from examples.common.functions import write_files
from runnable import Catalog, Job, PythonTask

print("Running catalog.py")


def main():
    write_catalog = Catalog(put=["df.csv", "data_folder/data.txt"])
    generate_data = PythonTask(
        name="generate_data",
        function=write_files,
        catalog=write_catalog,
    )

    job = Job(name="catalog", task=generate_data)

    _ = job.execute()

    return job


if __name__ == "__main__":
    main()
