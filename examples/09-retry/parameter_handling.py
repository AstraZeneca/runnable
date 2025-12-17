import os

from runnable import Pipeline, PythonTask, Stub


def increment_parameter(param1: int) -> int:
    return param1 + 1


def check_envvar(param1: int, param2: str):
    should_pass = os.environ.get("should_pass", "false").lower()

    if should_pass == "true":
        assert param1 == 42
        assert param2 == "hello world"
        return
    raise ValueError("Environment variable check failed!")


def main():
    increment_task = PythonTask(
        name="increment_parameter_task",
        function=increment_parameter,
        returns=["param1"],
    )

    can_fail_task = PythonTask(
        name="check_envvar_task",
        function=check_envvar,
    )

    pipeline = Pipeline(steps=[increment_task, can_fail_task])

    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    # Any parameter prefixed by "RUNNABLE_PRM_" will be picked up by runnable
    main()
