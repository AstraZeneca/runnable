import importlib
import os
from contextlib import contextmanager
from datetime import datetime
from functools import partial

import pytest

import tests.assertions as conditions
from runnable import defaults, names


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


def list_python_examples():
    for example in python_examples:
        yield example


@contextmanager
def runnable_context():
    from runnable import context as runnable_context

    os.environ.pop("RUNNABLE_CONFIGURATION_FILE", None)
    runnable_context.run_context = None
    try:
        yield runnable_context
    finally:
        os.environ.pop(defaults.ENV_RUN_ID, None)
        os.environ.pop("RUNNABLE_CONFIGURATION_FILE", None)
        os.environ.pop("RUNNABLE_PRM_envvar", None)
        print("Cleaning up runnable context")
        runnable_context.run_context = None


@contextmanager
def default_context():
    with runnable_context():
        os.environ["RUNNABLE_PRM_envvar"] = "from env"
        yield


@contextmanager
def emulator_context():
    with runnable_context():
        os.environ["RUNNABLE_CONFIGURATION_FILE"] = "examples/11-jobs/emulate.yaml"
        os.environ["RUNNABLE_PRM_envvar"] = "from env"
        yield


@contextmanager
def minio_context():
    with runnable_context():
        os.environ["RUNNABLE_CONFIGURATION_FILE"] = "examples/configs/minio.yaml"
        os.environ["RUNNABLE_PRM_envvar"] = "from env"
        yield


contexts = [default_context, emulator_context]


# file, parameters_file, assertions
python_examples = [
    (
        "11-jobs/python_tasks",
        "",
        [
            partial(conditions.should_have_job_and_status),
            partial(conditions.should_job_have_output_parameters, {}),
        ],
    ),
    (
        "11-jobs/scripts",
        "",
        [
            partial(conditions.should_have_job_and_status),
            partial(conditions.should_job_have_output_parameters, {}),
        ],
    ),
    (
        "11-jobs/catalog",
        "",
        [
            partial(conditions.should_have_job_and_status),
            partial(conditions.should_job_have_output_parameters, {}),
        ],
    ),
    (
        "11-jobs/catalog_no_copy",
        "",
        [
            partial(conditions.should_have_job_and_status),
            partial(conditions.should_job_have_output_parameters, {}),
        ],
    ),
    (
        "11-jobs/passing_parameters_python",
        "",
        [
            partial(conditions.should_have_job_and_status),
            partial(
                conditions.should_job_have_output_parameters,
                {
                    "integer": 1,
                    "floater": 3.14,
                    "stringer": "hello",
                    "pydantic_param": {"x": 10, "foo": "bar"},
                    "score": 0.9,
                    "df": "df",
                },
            ),
        ],
    ),
    (
        "11-jobs/notebooks",
        "",
        [
            partial(conditions.should_have_job_and_status),
            partial(conditions.should_job_have_output_parameters, {}),
        ],
    ),
]


@pytest.mark.parametrize("example", list_python_examples())
@pytest.mark.parametrize("context", contexts)
# @pytest.mark.no_cover
@pytest.mark.e2e
def test_python_examples(example, context):
    print(f"Testing {example}...")

    mod, _, assertions = example

    context = context()

    imported_module = importlib.import_module(f"examples.{mod.replace('/', '.')}")
    f = getattr(imported_module, "main")

    with context:
        from runnable import exceptions

        try:
            os.environ[defaults.ENV_RUN_ID] = generate_run_id()
            f()
        except exceptions.ExecutionFailedError:
            print("Example failed")

        [asserttion() for asserttion in assertions]


minio_contexts = [minio_context]


@pytest.mark.parametrize("example", list_python_examples())
@pytest.mark.parametrize("context", minio_contexts)
@pytest.mark.minio
@pytest.mark.e2e
def test_python_examples_minio(example, context):
    print(f"Testing {example}...")

    mod, _, assertions = example

    context = context()

    imported_module = importlib.import_module(f"examples.{mod.replace('/', '.')}")
    f = getattr(imported_module, "main")

    with context:
        from runnable import exceptions

        try:
            os.environ[defaults.ENV_RUN_ID] = generate_run_id()
            f()
        except exceptions.ExecutionFailedError:
            print("Example failed")

        [asserttion() for asserttion in assertions]
