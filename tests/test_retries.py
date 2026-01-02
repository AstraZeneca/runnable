import importlib
import os
from contextlib import contextmanager
from datetime import datetime

import pytest
from rich import print

from runnable import defaults, exceptions, names


def get_step(step_name: str, run_log):
    step, _ = run_log.search_step_by_internal_name(step_name)
    return step


def load_run_log(run_id):
    from runnable import context

    current_context = context.get_run_context()
    if current_context is None:
        raise RuntimeError("No run context available")
    run_log = current_context.run_log_store.get_run_log_by_id(run_id, full=True)
    return run_log


def generate_run_id(run_id: str = "") -> str:
    """Generate a new run_id.

    If the input run_id is none, we create one based on time stamp.

    Args:
        run_id (str, optional): Input Run ID. Defaults to None

    Returns:
        str: A generated run_id
    """
    # If we are not provided with a run_id, check env var
    if not run_id:
        run_id = os.environ.get(defaults.ENV_RUN_ID, "")

    # If both are not given, generate one
    if not run_id:
        now = datetime.now()
        run_id = f"{names.get_random_name()}-{now.hour:02}{now.minute:02}"

    return run_id


@contextmanager
def retry_context():
    from runnable import context as runnable_context

    try:
        os.environ.pop("RUNNABLE_RETRY_RUN_ID", None)
        os.environ.pop("RUNNABLE_RUN_ID", None)
        os.environ.pop("RUNNABLE_CONFIGURATION_FILE", None)
        os.environ.pop("RUNNABLE_PARAMETERS_FILE", None)
        os.environ.pop("RUNNABLE_PRM_param1", None)
        yield
    finally:
        os.environ.pop("RUNNABLE_RETRY_RUN_ID", None)
        os.environ.pop("RUNNABLE_RUN_ID", None)
        os.environ.pop("RUNNABLE_PRM_param1", None)
        os.environ.pop("RUNNABLE_PARAMETERS_FILE", None)
        print("Cleaning up runnable context")
        runnable_context.set_run_context(None)


@contextmanager
def chunked_fs_context():
    with retry_context():
        os.environ["RUNNABLE_CONFIGURATION_FILE"] = (
            "examples/configs/chunked-fs-run_log.yaml"
        )
        yield


contexts = [
    retry_context,
    chunked_fs_context,
    # mocked_context,
]

# module, step name, os_variables_first, os_variables_retry
examples = [
    (
        "examples/09-retry/simple_task",
        "check_envvar_task",
        {},
        {},
    ),
    (
        "examples/09-retry/linear",
        "check_envvar_task",
        {},
        {},
    ),
    (
        "examples/09-retry/parameter_handling",
        "check_envvar_task",
        {"RUNNABLE_PARAMETERS_FILE": "examples/09-retry/original_parameters.yaml"},
        {},
    ),
    (
        "examples/09-retry/parameter_handling",
        "check_envvar_task",
        {"RUNNABLE_PARAMETERS_FILE": "examples/09-retry/original_parameters.yaml"},
        {
            "RUNNABLE_PRM_param1": "43",
        },
    ),
    (
        "examples/09-retry/parallel",
        "parallel_step.branch1.hello python",
        {},
        {},
    ),
    (
        "examples/09-retry/parallel",
        "parallel_step.branch2.hello python",
        {},
        {},
    ),
    (
        "examples/09-retry/map",
        "map_state.1.execute_python",
        {"RUNNABLE_PARAMETERS_FILE": "examples/09-retry/original_parameters.yaml"},
        {},
    ),
]


@pytest.mark.parametrize("example", examples)
@pytest.mark.parametrize("context", contexts)
def test_examples(example, context):
    module, step_name, os_variables_first, os_variables_retry = example
    print(f"Testing module: {module} step: {step_name}")

    imported_module = importlib.import_module(f"{module.replace('/', '.')}")
    f = getattr(imported_module, "main")
    run_id = generate_run_id()

    with context():
        # First run should fail
        os.environ["should_pass"] = "false"
        os.environ["RUNNABLE_RUN_ID"] = run_id
        for k, v in os_variables_first.items():
            os.environ[k] = v
        try:
            f()
            assert False, "Expected ExecutionFailedError"
        except exceptions.ExecutionFailedError:
            run_log = load_run_log(run_id)
            step_log = get_step(step_name, run_log)
            assert len(step_log.attempts) == 1
            assert run_log.status == "FAIL"
            assert True

    with context():
        # Retry run should pass
        os.environ["should_pass"] = "true"
        os.environ["RUNNABLE_RETRY_RUN_ID"] = run_id
        for k, v in os_variables_retry.items():
            os.environ[k] = v
        try:
            f()
            run_log = load_run_log(run_id)
            step_log = get_step(step_name, run_log)
            assert len(step_log.attempts) == 2
            assert run_log.status == "SUCCESS"
            assert True
        except exceptions.ExecutionFailedError:
            assert False, "Retry run should have passed"
