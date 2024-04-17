from examples.common.functions import read_unpickled_parameter
from runnable import Pipeline, PythonTask, ShellTask, metric


def main():
    export_env_command = """
    export integer=1
    export floater=3.14
    export stringer="hello"
    export pydantic_param='{"x": 10, "foo": "bar"}'
    export score=0.9
    """
    write_parameters_in_shell = ShellTask(
        command=export_env_command,
        returns=[
            "integer",
            "floater",
            "stringer",
            "pydantic_param",
            metric("score"),
        ],
        name="write_parameter",
    )

    read_parameters = PythonTask(
        function=read_unpickled_parameter,
        name="read_parameters",
        terminate_with_success=True,
    )

    pipeline = Pipeline(
        steps=[write_parameters_in_shell, read_parameters],
    )

    _ = pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
