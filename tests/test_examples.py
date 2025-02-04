import importlib
import os
import subprocess
from contextlib import contextmanager
from functools import partial

import pytest

import tests.assertions as conditions
from runnable import defaults
from runnable.utils import generate_run_id


def list_python_examples():
    for example in python_examples:
        yield example


@contextmanager
def runnable_context():
    from runnable import context as runnable_context

    try:
        yield
    finally:
        del os.environ[defaults.ENV_RUN_ID]
        runnable_context.run_context = None


@contextmanager
def container_context():
    with runnable_context():
        os.environ["RUNNABLE_CONFIGURATION_FILE"] = (
            "examples/configs/local-container.yaml"
        )
        os.environ["RUNNABLE_PRM_envvar"] = "from env"
        yield
        del os.environ["RUNNABLE_CONFIGURATION_FILE"]
        del os.environ["RUNNABLE_PRM_envvar"]


@contextmanager
def minio_context():
    with runnable_context():
        os.environ["RUNNABLE_CONFIGURATION_FILE"] = "examples/configs/minio.yaml"
        os.environ["RUNNABLE_PRM_envvar"] = "from env"
        yield
        del os.environ["RUNNABLE_CONFIGURATION_FILE"]
        del os.environ["RUNNABLE_PRM_envvar"]


@contextmanager
def chunked_minio_context():
    with runnable_context():
        os.environ["RUNNABLE_CONFIGURATION_FILE"] = (
            "examples/configs/chunked_minio.yaml"
        )
        os.environ["RUNNABLE_PRM_envvar"] = "from env"
        yield
        del os.environ["RUNNABLE_CONFIGURATION_FILE"]
        del os.environ["RUNNABLE_PRM_envvar"]


@contextmanager
def chunked_fs_context():
    with runnable_context():
        os.environ["RUNNABLE_CONFIGURATION_FILE"] = (
            "examples/configs/chunked-fs-run_log.yaml"
        )
        os.environ["RUNNABLE_PRM_envvar"] = "from env"
        yield
        del os.environ["RUNNABLE_CONFIGURATION_FILE"]
        del os.environ["RUNNABLE_PRM_envvar"]


@contextmanager
def mocked_context():
    with runnable_context():
        os.environ["RUNNABLE_CONFIGURATION_FILE"] = (
            "examples/08-mocking/mocked-config-simple.yaml"
        )
        os.environ["RUNNABLE_PRM_envvar"] = "from env"
        yield
        del os.environ["RUNNABLE_CONFIGURATION_FILE"]
        del os.environ["RUNNABLE_PRM_envvar"]


@contextmanager
def patched_context():
    with runnable_context():
        os.environ["RUNNABLE_CONFIGURATION_FILE"] = "examples/08-mocking/patching.yaml"
        os.environ["RUNNABLE_PRM_envvar"] = "from env"
        yield
        del os.environ["RUNNABLE_CONFIGURATION_FILE"]
        del os.environ["RUNNABLE_PRM_envvar"]


@contextmanager
def default_context():
    with runnable_context():
        os.environ["RUNNABLE_PRM_envvar"] = "from env"
        yield
        del os.environ["RUNNABLE_PRM_envvar"]


@contextmanager
def argo_context():
    with runnable_context():
        os.environ["RUNNABLE_CONFIGURATION_FILE"] = "examples/configs/argo-config.yaml"
        yield
        subprocess.run(["argo", "lint", "--offline", "argo-pipeline.yaml"], check=True)
        del os.environ["RUNNABLE_CONFIGURATION_FILE"]


contexts = [
    default_context,
    chunked_fs_context,
]  # , mocked_context, argo_context]

# file, fails, ignore_contexts, parameters_file, assertions
python_examples = [
    (
        "01-tasks/python_tasks",
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
        True,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 3),
            partial(conditions.should_have_catalog_execution_logs),
            partial(conditions.should_be_failed),
            partial(conditions.should_step_be_successful, "step 4"),
            partial(conditions.should_step_be_failed, "step 1"),
        ],
    ),
    (
        "02-sequential/on_failure_succeed",
        False,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 3),
            partial(conditions.should_have_catalog_execution_logs),
            partial(conditions.should_be_successful),
            partial(conditions.should_step_be_successful, "step 4"),
            partial(conditions.should_step_be_failed, "step 1"),
        ],
    ),
    (
        "03-parameters/static_parameters_python",
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
        "06-parallel/parallel",
        False,
        [],
        "",
        [
            partial(conditions.should_have_num_steps, 2),
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
        True,
        [],
        "examples/common/initial_parameters.yaml",
        [
            partial(conditions.should_have_num_steps, 3),
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

    mod, fails, ignore_contexts, _, assertions = example
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
            [asserttion() for asserttion in assertions]

        except exceptions.ExecutionFailedError:
            print("Example failed")
            if not fails:
                raise


@pytest.mark.parametrize("example", list_python_examples())
@pytest.mark.parametrize("context", contexts)
# @pytest.mark.no_cover
@pytest.mark.e2e
def test_yaml_examples(example, context):
    print(f"Testing {example}...")
    file, fails, ignore_contexts, parameters_file, assertions = example

    if context in ignore_contexts:
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
            [asserttion() for asserttion in assertions]
        except exceptions.ExecutionFailedError:
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


# @pytest.mark.parametrize("example", list_python_examples())
# @pytest.mark.container
# def test_python_examples_container(example):
#     print(f"Testing {example}...")

#     mod, fails, _ = example
#     context = container_context()

#     imported_module = importlib.import_module(f"examples.{mod.replace('/', '.')}")
#     f = getattr(imported_module, "main")
#     with context:
#         from runnable import context, exceptions

#         try:
#             f()
#         except exceptions.ExecutionFailedError:
#             print("Example failed")
#             if not fails:
#                 raise
#         finally:
#             context.run_context = None


# @pytest.mark.parametrize("example", list_python_examples())
# @pytest.mark.container
# def test_yaml_examples_container(example):
#     print(f"Testing {example}...")
#     file, fails, _ = example

#     context = container_context()

#     example_file = f"examples/{file}.yaml"
#     parameters_file = "examples/common/initial_parameters.yaml"

#     with context:
#         from runnable import exceptions
#         from runnable.entrypoints import execute_pipeline_yaml_spec

#         try:
#             execute_pipeline_yaml_spec(
#                 pipeline_file=example_file, parameters_file=parameters_file
#             )
#         except exceptions.ExecutionFailedError:
#             if not fails:
#                 raise


# TODO: add tests for jobs
