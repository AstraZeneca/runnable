import os
import re
import sys
from functools import lru_cache
from pathlib import Path

# Ensure the project root directory is in the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# We'll import defaults inside each function to avoid circular imports


@lru_cache
def load_run_log(run_id):
    from runnable import context

    current_context = context.get_run_context()
    if current_context is None:
        raise RuntimeError("No run context available")
    run_log = current_context.run_log_store.get_run_log_by_id(run_id, full=True)
    return run_log


def get_catalog_location(run_id):
    run_log = load_run_log(run_id=run_id)

    catalog_name = run_log.run_config["catalog"]["service_name"]
    if catalog_name == "file-system":
        return ".catalog"
    elif catalog_name == "minio":
        return "minio-runnable/catalog"

    raise Exception("Unknown catalog")


def should_be_successful():
    from runnable import defaults

    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    assert (
        run_log.status == defaults.SUCCESS
    ), f"Expected successful, got {run_log.status}"


def should_be_failed():
    from runnable import defaults

    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    assert run_log.status == defaults.FAIL, f"Expected successful, got {run_log.status}"


def should_have_job_and_status():
    from runnable import defaults

    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)

    assert run_log.job is not None, "Expected job to be set, but it is None"
    assert run_log.status == defaults.SUCCESS
    assert run_log.job.status == defaults.SUCCESS


def should_have_num_steps(num_steps: int) -> None:
    from runnable import defaults

    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    assert (
        len(run_log.steps) == num_steps
    ), f"Expected {num_steps} steps, got {len(run_log.steps)}"


def should_step_be_successful(step_name: str):
    from runnable import defaults

    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    step = run_log.steps[step_name]
    assert step.status == defaults.SUCCESS, f"Expected successful, got {step.status}"


def should_step_be_failed(step_name: str):
    from runnable import defaults

    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    step = run_log.steps[step_name]
    assert step.status == defaults.FAIL, f"Expected failed, got {step.status}"


def should_step_have_parameters(step_name: str, parameters: dict):
    from runnable import defaults

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
    from runnable import defaults

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
    from runnable import defaults

    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    step = run_log.steps[step_name]
    func_parameters = {
        parameter: value.value
        for parameter, value in step.attempts[0].output_parameters.items()
    }

    assert parameters == func_parameters


def should_job_have_output_parameters(parameters: dict):
    from runnable import defaults

    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    job = run_log.job
    func_parameters = {
        parameter: value.value
        for parameter, value in job.attempts[0].output_parameters.items()
    }

    assert parameters == func_parameters


def should_have_catalog_execution_logs():
    from runnable import defaults

    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)

    step_names = run_log.steps.keys()
    catalog_location = get_catalog_location(run_id=run_id)
    contents = os.listdir(f"{catalog_location}/{run_id}")

    for step_name in step_names:
        step = run_log.steps[step_name]
        if step.step_type == "parallel":
            continue  # Skip the root node of the parallel step
        logfile_name = "".join(x for x in step_name if x.isalnum())
        pattern = rf"{re.escape(logfile_name)}...\.execution\.log"

        assert any(
            re.search(pattern, s) for s in contents
        ), "No match found in the list."


def should_have_catalog_contents(files: list[str]):
    from runnable import defaults

    run_id = os.environ[defaults.ENV_RUN_ID]

    catalog_location = get_catalog_location(run_id=run_id)
    contents = os.listdir(f"{catalog_location}/{run_id}")

    for file_name in files or []:
        pattern = rf"{file_name}"

        assert any(
            re.search(pattern, s) for s in contents
        ), "No match found in the list."


# TODO: No copy assertion


def should_branch_have_steps(step_name, branch_name: str, num_steps: int):
    from runnable import defaults

    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    step = run_log.steps[step_name]
    branch = step.branches[f"{step_name}.{branch_name}"]

    assert len(branch.steps) == num_steps


def should_branch_be_successful(step_name, branch_name: str):
    from runnable import defaults

    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    step = run_log.steps[step_name]
    branch = step.branches[f"{step_name}.{branch_name}"]

    assert branch.status == defaults.SUCCESS


def should_branch_be_failed(step_name, branch_name: str):
    from runnable import defaults

    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)
    step = run_log.steps[step_name]
    branch = step.branches[f"{step_name}.{branch_name}"]

    assert branch.status == defaults.FAIL


def should_have_notebook_output(name: str):
    from runnable import defaults

    run_id = os.environ[defaults.ENV_RUN_ID]

    catalog_location = get_catalog_location(run_id=run_id)

    catalog_location = Path(f"{catalog_location}/{run_id}")
    path = catalog_location / name

    assert path.exists()


def should_have_root_parameters(parameters: dict):
    """Verify parameters exist at run log root level with expected values."""
    from runnable import defaults

    run_id = os.environ[defaults.ENV_RUN_ID]
    run_log = load_run_log(run_id)

    for param_name, expected_value in parameters.items():
        assert (
            param_name in run_log.parameters
        ), f"Parameter {param_name} not found in root parameters"
        actual_value = run_log.parameters[param_name].get_value()
        assert (
            actual_value == expected_value
        ), f"Expected {param_name}={expected_value}, got {actual_value}"
