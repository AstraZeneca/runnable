"""
The below example showcases setting up known initial parameters for a pipeline
of notebook and shell based commands.

The initial parameters as defined in the yaml file are:
    integer: 1
    floater : 3.14
    stringer : hello
    pydantic_param:
        x: 10
        foo: bar

runnable exposes the nested parameters as dictionary for notebook based tasks
and as a json string for the shell based tasks.

You can set the initial parameters from environment variables as well.
eg: Any environment variable prefixed by "RUNNABLE_PRM_" will be picked up by runnable


Run this pipeline as:
    python examples/03-parameters/static_parameters_non_python.py
"""

from runnable import NotebookTask, Pipeline, ShellTask


def main():
    read_params_in_notebook = NotebookTask(
        name="read_params_in_notebook",
        notebook="examples/common/read_parameters.ipynb",
    )

    shell_command = """
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
    read_params_in_shell = ShellTask(
        name="read_params_in_shell",
        command=shell_command,
        terminate_with_success=True,
    )

    pipeline = Pipeline(
        steps=[read_params_in_notebook, read_params_in_shell],
    )

    _ = pipeline.execute(parameters_file="examples/common/initial_parameters.yaml")

    return pipeline


if __name__ == "__main__":
    main()
