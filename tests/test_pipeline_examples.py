import importlib
import os
import subprocess
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

    os.environ["FIX_RANDOM_TOSS"] = "heads"

    try:
        yield runnable_context
    finally:
        os.environ.pop(defaults.ENV_RUN_ID, None)
        os.environ.pop("RUNNABLE_CONFIGURATION_FILE", None)
        os.environ.pop("RUNNABLE_PRM_envvar", None)
        os.environ.pop("FIX_RANDOM_TOSS", None)
        print("Cleaning up runnable context")
        runnable_context.run_context = None
        runnable_context.progress = None


@contextmanager
def minio_context():
    with runnable_context():
        os.environ["RUNNABLE_CONFIGURATION_FILE"] = "examples/configs/minio.yaml"
        os.environ["RUNNABLE_PRM_envvar"] = "from env"
        yield


@contextmanager
def chunked_minio_context():
    with runnable_context():
        os.environ["RUNNABLE_CONFIGURATION_FILE"] = (
            "examples/configs/chunked_minio.yaml"
        )
        os.environ["RUNNABLE_PRM_envvar"] = "from env"
        yield


@contextmanager
def chunked_fs_context():
    with runnable_context():
        os.environ["RUNNABLE_CONFIGURATION_FILE"] = (
            "examples/configs/chunked-fs-run_log.yaml"
        )
        os.environ["RUNNABLE_PRM_envvar"] = "from env"
        yield


@contextmanager
def mocked_context():
    with runnable_context():
        os.environ["RUNNABLE_CONFIGURATION_FILE"] = (
            "examples/08-mocking/mocked-config-simple.yaml"
        )
        os.environ["RUNNABLE_PRM_envvar"] = "from env"
        yield


@contextmanager
def patched_context():
    with runnable_context():
        os.environ["RUNNABLE_CONFIGURATION_FILE"] = "examples/08-mocking/patching.yaml"
        os.environ["RUNNABLE_PRM_envvar"] = "from env"
        yield


@contextmanager
def default_context():
    with runnable_context():
        os.environ["RUNNABLE_PRM_envvar"] = "from env"
        yield


@contextmanager
def emulator_context():
    with runnable_context():
        os.environ["RUNNABLE_CONFIGURATION_FILE"] = "examples/configs/emulate.yaml"
        os.environ["RUNNABLE_PRM_envvar"] = "from env"
        yield


@contextmanager
def argo_context():
    with runnable_context():
        os.environ["RUNNABLE_CONFIGURATION_FILE"] = "examples/configs/argo-config.yaml"
        yield
        subprocess.run(["argo", "lint", "--offline", "argo-pipeline.yaml"], check=True)


contexts = [
    default_context,
    emulator_context,
    chunked_fs_context,
    # mocked_context,
]

# file, no_yaml, fails, ignore_contexts, parameters_file, assertions
python_examples = [
    (
        "01-tasks/python_tasks",
        False,
        False,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 2),
            partial(conditions.should_have_catalog_execution_logs),
        ],
    ),
    (
        "01-tasks/python_task_as_pipeline",
        True,
        False,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 2),
            partial(conditions.should_have_catalog_execution_logs),
        ],
    ),
    (
        "01-tasks/scripts",
        False,
        False,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 2),
            partial(conditions.should_have_catalog_execution_logs),
        ],
    ),
    (
        "01-tasks/stub",
        False,
        False,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 4),
            partial(conditions.should_have_catalog_execution_logs),
        ],
    ),
    (
        "01-tasks/notebook",
        False,
        False,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 2),
            partial(conditions.should_have_catalog_execution_logs),
            partial(
                conditions.should_have_notebook_output,
                "examples/common/simple_notebook-hello_out.ipynb",
            ),
        ],
    ),
    (
        "02-sequential/traversal",
        False,
        False,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 5),
            partial(conditions.should_have_catalog_execution_logs),
            partial(conditions.should_be_successful),
            partial(conditions.should_step_be_successful, "hello stub"),
            partial(conditions.should_step_be_successful, "hello python"),
            partial(conditions.should_step_be_successful, "hello shell"),
            partial(conditions.should_step_be_successful, "hello notebook"),
        ],
    ),
    (
        "02-sequential/default_fail",
        False,
        True,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 3),
            partial(conditions.should_have_catalog_execution_logs),
            partial(conditions.should_be_failed),
            partial(conditions.should_step_be_successful, "step 1"),
            partial(conditions.should_step_be_failed, "step 2"),
        ],
    ),
    (
        "02-sequential/on_failure_fail",
        False,
        True,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 3),
            partial(conditions.should_have_catalog_execution_logs),
            partial(conditions.should_be_failed),
            partial(conditions.should_step_be_successful, "step_4"),
            partial(conditions.should_step_be_failed, "step_1"),
        ],
    ),
    (
        "02-sequential/on_failure_succeed",
        False,
        False,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 3),
            partial(conditions.should_have_catalog_execution_logs),
            partial(conditions.should_be_successful),
            partial(conditions.should_step_be_successful, "step_4"),
            partial(conditions.should_step_be_failed, "step_1"),
        ],
    ),
    (
        "02-sequential/conditional",
        True,
        False,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 4),
            # partial(conditions.should_have_catalog_execution_logs),
            partial(conditions.should_be_successful),
            partial(conditions.should_step_be_successful, "toss_task"),
            partial(conditions.should_step_be_successful, "conditional"),
            partial(conditions.should_branch_have_steps, "conditional", "heads", 2),
            # partial(conditions.should_branch_have_steps, "conditional", "tails", 1),
        ],
    ),
    (
        "03-parameters/static_parameters_python",
        False,
        False,
        [],
        "examples/common/initial_parameters.yaml",
        [
            partial(conditions.should_have_num_steps, 3),
            partial(conditions.should_have_catalog_execution_logs),
            partial(conditions.should_be_successful),
            partial(conditions.should_step_be_successful, "read_params_as_pydantic"),
            partial(conditions.should_step_be_successful, "read_params_as_json"),
            partial(
                conditions.should_step_have_parameters,
                "read_params_as_pydantic",
                {
                    "integer": 1,
                    "floater": 3.14,
                    "stringer": "hello",
                    "pydantic_param": {"x": 10, "foo": "bar"},
                    "chunks": [1, 2, 3],
                    "envvar": "from env",
                },
            ),
            partial(
                conditions.should_step_have_parameters,
                "read_params_as_json",
                {
                    "integer": 1,
                    "floater": 3.14,
                    "stringer": "hello",
                    "pydantic_param": {"x": 10, "foo": "bar"},
                    "chunks": [1, 2, 3],
                    "envvar": "from env",
                },
            ),
        ],
    ),
    (
        "03-parameters/static_parameters_non_python",
        False,
        False,
        [],
        "examples/common/initial_parameters.yaml",
        [
            partial(conditions.should_have_num_steps, 3),
            partial(conditions.should_have_catalog_execution_logs),
            partial(conditions.should_be_successful),
            partial(conditions.should_step_be_successful, "read_params_in_notebook"),
            partial(conditions.should_step_be_successful, "read_params_in_shell"),
            partial(
                conditions.should_step_have_parameters,
                "read_params_in_notebook",
                {
                    "integer": 1,
                    "floater": 3.14,
                    "stringer": "hello",
                    "pydantic_param": {"x": 10, "foo": "bar"},
                    "chunks": [1, 2, 3],
                    "envvar": "from env",
                },
            ),
            partial(
                conditions.should_step_have_parameters,
                "read_params_in_shell",
                {
                    "integer": 1,
                    "floater": 3.14,
                    "stringer": "hello",
                    "pydantic_param": {"x": 10, "foo": "bar"},
                    "chunks": [1, 2, 3],
                    "envvar": "from env",
                },
            ),
        ],
    ),
    (
        "03-parameters/static_parameters_fail",
        False,
        False,
        [],
        "examples/common/initial_parameters.yaml",
        [
            partial(conditions.should_have_num_steps, 3),
            partial(conditions.should_have_catalog_execution_logs),
            partial(conditions.should_be_successful),
            partial(conditions.should_step_be_failed, "read_params_and_fail"),
            partial(conditions.should_step_be_successful, "read_params_in_notebook"),
            partial(
                conditions.should_step_have_parameters,
                "read_params_in_notebook",
                {
                    "integer": 1,
                    "floater": 3.14,
                    "stringer": "hello",
                    "pydantic_param": {"x": 10, "foo": "bar"},
                    "chunks": [1, 2, 3],
                    "envvar": "from env",
                },
            ),
            partial(
                conditions.should_step_have_parameters,
                "read_params_and_fail",
                {
                    "integer": 1,
                    "floater": 3.14,
                    "stringer": "hello",
                    "pydantic_param": {"x": 10, "foo": "bar"},
                    "chunks": [1, 2, 3],
                    "envvar": "from env",
                },
            ),
        ],
    ),
    (
        "03-parameters/passing_parameters_python",
        False,
        False,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 3),
            partial(conditions.should_have_catalog_execution_logs),
            partial(conditions.should_be_successful),
            partial(conditions.should_step_be_successful, "set_parameter"),
            partial(
                conditions.should_step_have_parameters,
                "set_parameter",
                {
                    "envvar": "from env",
                },
            ),
            partial(
                conditions.should_step_have_output_parameters,
                "set_parameter",
                {
                    "integer": 1,
                    "floater": 3.14,
                    "stringer": "hello",
                    "pydantic_param": {"x": 10, "foo": "bar"},
                    "score": 0.9,
                    "df": "df",
                },
            ),
            partial(
                conditions.should_step_have_parameters,
                "get_parameters",
                {
                    "integer": 1,
                    "floater": 3.14,
                    "stringer": "hello",
                    "pydantic_param": {"x": 10, "foo": "bar"},
                    "score": 0.9,
                    "df": "df",
                    "envvar": "from env",
                },
            ),
            partial(
                conditions.should_step_have_output_parameters,
                "get_parameters",
                {},
            ),
        ],
    ),
    (
        "03-parameters/passing_parameters_notebook",
        False,
        False,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 4),
            partial(conditions.should_have_catalog_execution_logs),
            partial(conditions.should_be_successful),
            partial(conditions.should_step_be_successful, "set_parameter"),
            partial(
                conditions.should_step_have_parameters,
                "set_parameter",
                {
                    "envvar": "from env",
                },
            ),
            partial(
                conditions.should_step_have_output_parameters,
                "set_parameter",
                {
                    "integer": 1,
                    "floater": 3.14,
                    "stringer": "hello",
                    "pydantic_param": {"x": 10, "foo": "bar"},
                    "score": 0.9,
                    "df": "df",
                },
            ),
            partial(
                conditions.should_step_have_parameters,
                "get_parameters",
                {
                    "integer": 1,
                    "floater": 3.14,
                    "stringer": "hello",
                    "pydantic_param": {"x": 10, "foo": "bar"},
                    "score": 0.9,
                    "df": "df",
                    "envvar": "from env",
                },
            ),
            partial(
                conditions.should_step_have_output_parameters,
                "get_parameters",
                {},
            ),
            partial(
                conditions.should_step_have_parameters,
                "read_parameters_in_notebook",
                {
                    "integer": 1,
                    "floater": 3.14,
                    "stringer": "hello",
                    "pydantic_param": {"x": 10, "foo": "bar"},
                    "score": 0.9,
                    "envvar": "from env",
                },
            ),
            partial(
                conditions.should_step_have_output_parameters,
                "read_parameters_in_notebook",
                {},
            ),
        ],
    ),
    (
        "03-parameters/passing_parameters_shell",
        False,
        False,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 4),
            partial(conditions.should_have_catalog_execution_logs),
            partial(conditions.should_be_successful),
            partial(conditions.should_step_be_successful, "write_parameter"),
            partial(
                conditions.should_step_have_parameters,
                "write_parameter",
                {
                    "envvar": "from env",
                },
            ),
            partial(
                conditions.should_step_have_output_parameters,
                "write_parameter",
                {
                    "integer": 1,
                    "floater": 3.14,
                    "stringer": "hello",
                    "pydantic_param": {"x": 10, "foo": "bar"},
                    "score": 0.9,
                },
            ),
            partial(
                conditions.should_step_have_parameters,
                "read_parameters",
                {
                    "integer": 1,
                    "floater": 3.14,
                    "stringer": "hello",
                    "pydantic_param": {"x": 10, "foo": "bar"},
                    "score": 0.9,
                    "envvar": "from env",
                },
            ),
            partial(
                conditions.should_step_have_output_parameters,
                "read_parameters",
                {},
            ),
            partial(
                conditions.should_step_have_parameters,
                "read_parameters_in_shell",
                {
                    "integer": 1,
                    "floater": 3.14,
                    "stringer": "hello",
                    "pydantic_param": {"x": 10, "foo": "bar"},
                    "score": 0.9,
                    "envvar": "from env",
                },
            ),
            partial(
                conditions.should_step_have_output_parameters,
                "read_parameters_in_shell",
                {},
            ),
        ],
    ),
    (
        "04-catalog/catalog_python",
        False,
        False,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 4),
            partial(conditions.should_have_catalog_execution_logs),
            partial(conditions.should_be_successful),
            partial(conditions.should_step_be_successful, "generate_data_python"),
            partial(conditions.should_step_be_successful, "read_data_python"),
            partial(
                conditions.should_have_catalog_contents,
                ["df.csv", "data_folder"],
            ),
        ],
    ),
    (
        "04-catalog/catalog_no_copy",
        True,
        False,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 4),
            partial(conditions.should_have_catalog_execution_logs),
            partial(conditions.should_be_successful),
            partial(conditions.should_step_be_successful, "generate_data_python"),
            partial(conditions.should_step_be_successful, "check_files_do_not_exist"),
        ],
    ),
    (
        "04-catalog/catalog_on_fail",
        False,
        False,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 3),
            partial(conditions.should_have_catalog_execution_logs),
            partial(conditions.should_be_successful),
            partial(conditions.should_step_be_failed, "fail_immediately"),
            partial(
                conditions.should_have_catalog_contents,
                ["df.csv"],
            ),
        ],
    ),
    (
        "06-parallel/parallel",
        False,
        False,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 3),
            partial(conditions.should_be_successful),
            partial(conditions.should_have_catalog_execution_logs),
            partial(conditions.should_branch_have_steps, "parallel_step", "branch1", 5),
            partial(conditions.should_branch_have_steps, "parallel_step", "branch2", 5),
            partial(conditions.should_branch_be_successful, "parallel_step", "branch1"),
            partial(conditions.should_branch_be_successful, "parallel_step", "branch2"),
        ],
    ),
    (
        "06-parallel/parallel_branch_fail",
        False,
        True,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 2),
            partial(conditions.should_be_failed),
            partial(conditions.should_have_catalog_execution_logs),
            partial(conditions.should_branch_have_steps, "parallel_step", "branch1", 5),
            partial(conditions.should_branch_have_steps, "parallel_step", "branch2", 3),
            partial(conditions.should_branch_be_successful, "parallel_step", "branch1"),
            partial(conditions.should_branch_be_failed, "parallel_step", "branch2"),
        ],
    ),
    (
        "07-map/map",
        False,
        False,
        [],
        "examples/common/initial_parameters.yaml",
        [
            partial(conditions.should_have_num_steps, 3),
            partial(conditions.should_be_successful),
            partial(conditions.should_branch_have_steps, "map_state", "1", 5),
            partial(conditions.should_branch_have_steps, "map_state", "2", 5),
            partial(conditions.should_branch_have_steps, "map_state", "3", 5),
            partial(
                conditions.should_branch_step_have_parameters,
                "map_state",
                "1",
                "execute_python",
                key="chunk",
                value=1,
            ),
            partial(
                conditions.should_branch_step_have_parameters,
                "map_state",
                "1",
                "execute_notebook",
                key="processed_python",
                value=10,
            ),
            partial(
                conditions.should_branch_step_have_parameters,
                "map_state",
                "1",
                "execute_shell",
                key="processed_notebook",
                value=100,
            ),
        ],
    ),
    (
        "07-map/map_fail",
        False,
        True,
        [],
        "examples/common/initial_parameters.yaml",
        [
            partial(conditions.should_have_num_steps, 2),
            partial(conditions.should_be_failed),
            partial(conditions.should_branch_have_steps, "map_state", "1", 2),
            partial(conditions.should_branch_have_steps, "map_state", "2", 5),
            partial(conditions.should_branch_have_steps, "map_state", "3", 5),
            partial(conditions.should_branch_be_failed, "map_state", "1"),
            partial(conditions.should_branch_be_successful, "map_state", "2"),
            partial(conditions.should_branch_be_successful, "map_state", "3"),
            partial(
                conditions.should_branch_step_have_parameters,
                "map_state",
                "1",
                "execute_python",
                key="chunk",
                value=1,
            ),
            partial(
                conditions.should_branch_step_have_parameters,
                "map_state",
                "2",
                "execute_notebook",
                key="processed_python",
                value=20,
            ),
            partial(
                conditions.should_branch_step_have_parameters,
                "map_state",
                "2",
                "execute_shell",
                key="processed_notebook",
                value=200,
            ),
        ],
    ),
]


@pytest.mark.parametrize("example", list_python_examples())
@pytest.mark.parametrize("context", contexts)
# @pytest.mark.no_cover
@pytest.mark.e2e
def test_python_examples(example, context):
    print(f"Testing {example}...")

    mod, _, fails, ignore_contexts, _, assertions = example
    if context in ignore_contexts:
        return

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
            if not fails:
                raise
        [asserttion() for asserttion in assertions]


@pytest.mark.parametrize("example", list_python_examples())
@pytest.mark.parametrize("context", contexts)
# @pytest.mark.no_cover
@pytest.mark.e2e
def test_yaml_examples(example, context):
    print(f"Testing {example}...")
    file, no_yaml, fails, ignore_contexts, parameters_file, assertions = example

    if context in ignore_contexts:
        return

    if no_yaml:
        return

    context = context()
    example_file = f"examples/{file}.yaml"

    with context:
        from runnable import exceptions
        from runnable.entrypoints import execute_pipeline_yaml_spec

        run_id = generate_run_id()
        os.environ[defaults.ENV_RUN_ID] = run_id
        try:
            execute_pipeline_yaml_spec(
                pipeline_file=example_file,
                parameters_file=parameters_file,
                run_id=os.environ[defaults.ENV_RUN_ID],
            )
        except exceptions.ExecutionFailedError:
            if not fails:
                raise
        [assertion() for assertion in assertions]


argo_contexts = [argo_context]


@pytest.mark.parametrize("example", list_python_examples())
@pytest.mark.parametrize("context", argo_contexts)
@pytest.mark.argo
@pytest.mark.e2e
def test_python_examples_argo(example, context):
    print(f"Testing {example}...")

    mod, _, fails, ignore_contexts, _, _ = example

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
            if not fails:
                raise


minio_contexts = [minio_context, chunked_minio_context]


@pytest.mark.parametrize("example", list_python_examples())
@pytest.mark.parametrize("context", minio_contexts)
@pytest.mark.minio
@pytest.mark.e2e
def test_python_examples_minio(example, context):
    print(f"Testing {example}...")

    mod, fails, _, _, assertions = example

    context = context()

    imported_module = importlib.import_module(f"examples.{mod.replace('/', '.')}")
    f = getattr(imported_module, "main")

    with context:
        from runnable import exceptions

        try:
            os.environ[defaults.ENV_RUN_ID] = generate_run_id()
            f()
            # [asserttion() for asserttion in assertions]

        except exceptions.ExecutionFailedError:
            print("Example failed")
            if not fails:
                raise


# TODO: add tests for jobs
