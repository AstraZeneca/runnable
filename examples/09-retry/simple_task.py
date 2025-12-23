import os

from runnable import PythonTask


def check_envvar():
    should_pass = os.environ.get("should_pass", "false").lower()

    if should_pass == "true":
        return
    raise ValueError("Environment variable check failed!")


def main():
    can_fail_pipeline = PythonTask(
        name="check_envvar_task",
        function=check_envvar,
    ).as_pipeline()

    can_fail_pipeline.execute()

    return can_fail_pipeline


if __name__ == "__main__":
    # Any parameter prefixed by "RUNNABLE_PRM_" will be picked up by runnable
    main()
