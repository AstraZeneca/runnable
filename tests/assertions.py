import json
import os
import re
from functools import lru_cache
from pathlib import Path

from runnable import defaults
from runnable.datastore import RunLog, StepLog


def gather_steps_from_step(step_log: StepLog) -> list[str]:
    steps = []
    for _, branch in step_log.branches.items():
        for _, step in branch.steps.items():
            if step.step_type == "task":
                steps.append(step.internal_name)
                continue

            steps.extend(gather_steps_from_step(step))
    return steps


def gather_steps(run_log: RunLog) -> list[str]:
    steps = []
    for _, step in run_log.steps.items():
        if step.step_type == "task":
            steps.append(step.internal_name)
            continue
        steps.extend(gather_steps_from_step(step))

    return steps


@lru_cache
def load_run_log(run_id):
    run_log_locaton = f".run_log_store/{run_id}.json"
    with open(run_log_locaton, "r") as f:
        run_log_dict = json.load(f)

    return RunLog(**run_log_dict)


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
    assert step.status == defaults.FAIL, f"Expected successful, got {step.status}"


def should_step_have_parameters(step_name: str, parameters: dict):
    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    step = run_log.steps[step_name]
    func_parameters = {
        parameter: value.value
        for parameter, value in step.attempts[0].input_parameters.items()
    }
    print(parameters)
    print(func_parameters)
    assert parameters == func_parameters


def should_have_catalog_execution_logs():
    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)

    step_names = gather_steps(run_log)
    contents = os.listdir(f".catalog/{run_id}")

    for step_name in step_names:
        logfile_name = "".join(x for x in step_name if x.isalnum())
        pattern = rf"{re.escape(logfile_name)}...\.execution\.log"

        print(f"Checking for {pattern} in {contents}")

        assert any(
            re.search(pattern, s) for s in contents
        ), "No match found in the list."


def should_have_notebook_output(name: str):
    run_id = os.environ[defaults.ENV_RUN_ID]

    catalog_location = Path(f".catalog/{run_id}")
    path = catalog_location / name

    assert path.is_file()
