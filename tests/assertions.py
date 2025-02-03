import os
import re
from functools import lru_cache
from pathlib import Path

from runnable import defaults


@lru_cache
def load_run_log(run_id):
    from runnable import context

    run_log = context.run_context.run_log_store.get_run_log_by_id(run_id, full=True)
    return run_log


def should_be_successful():
    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    assert (
        run_log.status == defaults.SUCCESS
    ), f"Expected successful, got {run_log.status}"


def should_be_failed():
    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    assert run_log.status == defaults.FAIL, f"Expected successful, got {run_log.status}"


def should_have_num_steps(num_steps: int) -> None:
    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    assert (
        len(run_log.steps) == num_steps
    ), f"Expected {num_steps} steps, got {len(run_log.steps)}"


def should_step_be_successful(step_name: str):
    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    step = run_log.steps[step_name]
    assert step.status == defaults.SUCCESS, f"Expected successful, got {step.status}"


def should_step_be_failed(step_name: str):
    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    step = run_log.steps[step_name]
    assert step.status == defaults.FAIL, f"Expected failed, got {step.status}"


def should_step_have_parameters(step_name: str, parameters: dict):
    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    step = run_log.steps[step_name]
    func_parameters = {
        parameter: value.value
        for parameter, value in step.attempts[0].input_parameters.items()
    }

    assert parameters == func_parameters


def should_branch_step_have_parameters(
    parent_step_name: str, branch_name: str, branch_step_name: str, key: str, value
):
    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    step = run_log.steps[parent_step_name]

    branch = step.branches[f"{parent_step_name}.{branch_name}"]
    step_name = f"{parent_step_name}.{branch_name}.{branch_step_name}"
    func_parameters = {
        parameter: value.value
        for parameter, value in branch.steps[step_name]
        .attempts[0]
        .input_parameters.items()
    }

    assert func_parameters[key] == value


def should_step_have_output_parameters(step_name: str, parameters: dict):
    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    step = run_log.steps[step_name]
    func_parameters = {
        parameter: value.value
        for parameter, value in step.attempts[0].output_parameters.items()
    }

    assert parameters == func_parameters


def should_have_catalog_execution_logs():
    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)

    step_names = run_log.steps.keys()
    contents = os.listdir(f".catalog/{run_id}")

    for step_name in step_names:
        logfile_name = "".join(x for x in step_name if x.isalnum())
        pattern = rf"{re.escape(logfile_name)}...\.execution\.log"

        assert any(
            re.search(pattern, s) for s in contents
        ), "No match found in the list."


def should_have_catalog_contents(files: list[str]):
    run_id = os.environ[defaults.ENV_RUN_ID]

    contents = os.listdir(f".catalog/{run_id}")

    for file_name in files or []:
        pattern = rf"{file_name}"

        assert any(
            re.search(pattern, s) for s in contents
        ), "No match found in the list."


def should_branch_have_steps(step_name, branch_name: str, num_steps: int):
    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    step = run_log.steps[step_name]
    branch = step.branches[f"{step_name}.{branch_name}"]

    assert len(branch.steps) == num_steps


def should_branch_be_successful(step_name, branch_name: str):
    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    step = run_log.steps[step_name]
    branch = step.branches[f"{step_name}.{branch_name}"]

    assert branch.status == defaults.SUCCESS


def should_branch_be_failed(step_name, branch_name: str):
    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    step = run_log.steps[step_name]
    branch = step.branches[f"{step_name}.{branch_name}"]

    assert branch.status == defaults.FAIL


def should_have_notebook_output(name: str):
    run_id = os.environ[defaults.ENV_RUN_ID]

    catalog_location = Path(f".catalog/{run_id}")
    path = catalog_location / name

    assert path.is_file()
