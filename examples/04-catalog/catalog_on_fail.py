"""
If a task fails, the put action of catalog will ignore files that are not present.

"""

from runnable import Catalog, Pipeline, ShellTask


def main():
    write_catalog = Catalog(put=["df.csv", "data_folder/data.txt"])
    fail_immediately = ShellTask(
        name="fail_immediately",
        command="""
        touch df.csv && \
        exit 1""",
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
