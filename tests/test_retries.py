import importlib
import os
from contextlib import contextmanager
from datetime import datetime

from rich import print

from runnable import defaults, exceptions, names


def get_step(step_name: str, run_log):
    step = run_log.steps[step_name]
    return step


def load_run_log(run_id):
    from runnable import context

    run_log = context.run_context.run_log_store.get_run_log_by_id(run_id, full=True)
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
        yield
    finally:
        os.environ.pop("RUNNABLE_RETRY_RUN_ID", None)
        os.environ.pop("RUNNABLE_RUN_ID", None)
        os.environ.pop("RUNNABLE_PRM_param1", None)
        print("Cleaning up runnable context")
        runnable_context.run_context = None


def test_simple_task():
    module = "examples/09-retry/simple_task"
    imported_module = importlib.import_module(f"{module.replace('/', '.')}")

    f = getattr(imported_module, "main")

    run_id = generate_run_id()

    with retry_context():
        # First run should fail
        os.environ["should_pass"] = "false"
        os.environ["RUNNABLE_RUN_ID"] = run_id
        try:
            f()
            assert False, "Expected ExecutionFailedError"
        except exceptions.ExecutionFailedError:
            run_log = load_run_log(run_id)
            step_log = get_step("check_envvar_task", run_log)
            assert len(step_log.attempts) == 1
            assert run_log.status == "FAIL"
            assert True

    with retry_context():
        # Retry run should pass
        os.environ["should_pass"] = "true"
        os.environ["RUNNABLE_RETRY_RUN_ID"] = run_id
        try:
            f()
            run_log = load_run_log(run_id)
            step_log = get_step("check_envvar_task", run_log)
            assert len(step_log.attempts) == 2
            assert run_log.status == "SUCCESS"
            assert True
        except exceptions.ExecutionFailedError:
            assert False, "Retry run should have passed"


def test_linear():
    module = "examples/09-retry/linear"
    imported_module = importlib.import_module(f"{module.replace('/', '.')}")

    f = getattr(imported_module, "main")

    run_id = generate_run_id()

    with retry_context():
        # First run should fail
        os.environ["should_pass"] = "false"
        os.environ["RUNNABLE_RUN_ID"] = run_id
        try:
            f()
            assert False, "Expected ExecutionFailedError"
        except exceptions.ExecutionFailedError:
            run_log = load_run_log(run_id)
            step_log = get_step("check_envvar_task", run_log)
            assert len(step_log.attempts) == 1
            assert run_log.status == "FAIL"
            assert True

    with retry_context():
        # Retry run should pass
        os.environ["should_pass"] = "true"
        os.environ["RUNNABLE_RETRY_RUN_ID"] = run_id
        try:
            f()
            run_log = load_run_log(run_id)
            step_log = get_step("check_envvar_task", run_log)
            assert len(step_log.attempts) == 2
            assert run_log.status == "SUCCESS"
            assert True
        except exceptions.ExecutionFailedError:
            assert False, "Retry run should have passed"


def test_parameter_handling():
    module = "examples/09-retry/parameter_handling"
    imported_module = importlib.import_module(f"{module.replace('/', '.')}")

    f = getattr(imported_module, "main")

    run_id = generate_run_id()

    with retry_context():
        # First run should fail
        os.environ["should_pass"] = "false"
        os.environ["RUNNABLE_RUN_ID"] = run_id
        os.environ["RUNNABLE_PARAMETERS_FILE"] = (
            "examples/09-retry/original_parameters.yaml"
        )
        try:
            f()
            assert False, "Expected ExecutionFailedError"
        except exceptions.ExecutionFailedError:
            run_log = load_run_log(run_id)
            step_log = get_step("check_envvar_task", run_log)
            assert len(step_log.attempts) == 1
            assert run_log.status == "FAIL"
            assert True

    with retry_context():
        # Retry run should pass
        os.environ["should_pass"] = "true"
        os.environ["RUNNABLE_RETRY_RUN_ID"] = run_id
        try:
            f()
            run_log = load_run_log(run_id)
            step_log = get_step("check_envvar_task", run_log)
            assert len(step_log.attempts) == 2
            assert run_log.status == "SUCCESS"
            assert True
        except exceptions.ExecutionFailedError:
            assert False, "Retry run should have passed"

    with retry_context():
        # Retry run should pass
        os.environ["should_pass"] = "true"
        os.environ["RUNNABLE_RETRY_RUN_ID"] = run_id
        os.environ["RUNNABLE_PRM_param1"] = "43"
        try:
            f()
            run_log = load_run_log(run_id)
            step_log = get_step("check_envvar_task", run_log)
            assert len(step_log.attempts) == 2
            assert run_log.status == "SUCCESS"
            assert True
        except exceptions.ExecutionFailedError:
            assert False, "Retry run should have passed"


def test_parameter_handling_environment():
    module = "examples/09-retry/parameter_handling"
    imported_module = importlib.import_module(f"{module.replace('/', '.')}")

    f = getattr(imported_module, "main")

    run_id = generate_run_id()

    with retry_context():
        # First run should fail
        os.environ["should_pass"] = "false"
        os.environ["RUNNABLE_RUN_ID"] = run_id
        os.environ["RUNNABLE_PARAMETERS_FILE"] = (
            "examples/09-retry/original_parameters.yaml"
        )
        try:
            f()
            assert False, "Expected ExecutionFailedError"
        except exceptions.ExecutionFailedError:
            run_log = load_run_log(run_id)
            step_log = get_step("check_envvar_task", run_log)
            assert len(step_log.attempts) == 1
            assert run_log.status == "FAIL"
            assert True

    with retry_context():
        # Retry run should pass
        os.environ["should_pass"] = "true"
        os.environ["RUNNABLE_RETRY_RUN_ID"] = run_id
        os.environ["RUNNABLE_PRM_param1"] = "43"
        try:
            f()
            run_log = load_run_log(run_id)
            step_log = get_step("check_envvar_task", run_log)
            assert len(step_log.attempts) == 2
            assert run_log.status == "SUCCESS"
            assert True
        except exceptions.ExecutionFailedError:
            assert False, "Retry run should have passed"


def test_conditional():
    module = "examples/09-retry/conditional"
    imported_module = importlib.import_module(f"{module.replace('/', '.')}")

    f = getattr(imported_module, "main")

    run_id = generate_run_id()

    with retry_context():
        # First run should fail
        os.environ["should_pass"] = "false"
        os.environ["RUNNABLE_RUN_ID"] = run_id
        os.environ["FIX_RANDOM_TOSS"] = "heads"
        try:
            f()
            assert False, "Expected ExecutionFailedError"
        except exceptions.ExecutionFailedError:
            run_log = load_run_log(run_id)
            # step_log = get_step("check_envvar_task", run_log)
            # assert len(step_log.attempts) == 1
            assert run_log.status == "FAIL"
            assert True

    with retry_context():
        # Retry run should pass
        os.environ["should_pass"] = "true"
        os.environ["RUNNABLE_RETRY_RUN_ID"] = run_id
        os.environ["FIX_RANDOM_TOSS"] = "heads"
        try:
            f()
            run_log = load_run_log(run_id)
            conditional_step_log = get_step("conditional", run_log)
            step_log = conditional_step_log.branches["conditional.heads"].steps[
                "conditional.heads.check_envvar"
            ]
            assert len(step_log.attempts) == 2
            assert run_log.status == "SUCCESS"
            assert False
        except exceptions.ExecutionFailedError:
            assert False, "Retry run should have passed"
