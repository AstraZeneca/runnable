"""
Demonstrates passing parameters to and from shell scripts.

We can extract only JSON serializable parameters from shell scripts.
eg: write_parameters_in_shell

We can only read json style parameters from shell scripts.
eg: read_parameters_in_shell
pydantic parameters are injected as json.

Run the below example as:
    python examples/03-parameters/passing_parameters_shell.py

"""

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
    )

    read_parameters_command = """
    if [ "$integer" = 1 ] \
        && [ "$floater" = 3.14 ] \
        && [ "$stringer" = "hello" ] \
        && [ "$pydantic_param" = '{"x": 10, "foo": "bar"}' ]; then
            echo "yaay"
            exit 0;
        else
            echo "naay"
            exit 1;
    fi
    """
    read_parameters_in_shell = ShellTask(
        name="read_parameters_in_shell",
        command=read_parameters_command,
        terminate_with_success=True,
    )

    pipeline = Pipeline(
        steps=[write_parameters_in_shell, read_parameters, read_parameters_in_shell],
    )

    _ = pipeline.execute()

    return pipeline


if __name__ == "__main__":
    main()
