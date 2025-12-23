import os

from runnable import Pipeline, PythonTask, Stub


def check_envvar():
    print(os.environ)
    # return
    raise Exception("Simulated failure for retry demonstration")
    # should_pass = os.environ.get("should_pass", "false").lower()

    # if should_pass == "true":
    #     return
    # raise ValueError("Environment variable check failed!")


def main():
    stub_node = Stub(name="stub_node")

    can_fail_task = PythonTask(
        name="check_envvar_task",
        function=check_envvar,
    )

    pipeline = Pipeline(steps=[stub_node, can_fail_task])

    pipeline.execute()

    return pipeline


if __name__ == "__main__":
    # Any parameter prefixed by "RUNNABLE_PRM_" will be picked up by runnable
    main()
