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
        os.environ["RUNNABLE_CONFIGURATION_FILE"] = "examples/08-mocking/default.yaml"
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
    mocked_context,
]  # , chunked_fs_context, mocked_context, argo_context]

python_examples = [
    (
        "01-tasks/python_tasks",
        False,
        [],
        [
            partial(conditions.should_have_num_steps, 2),
            partial(conditions.should_have_catalog_execution_logs),
        ],
    ),
    (
        "01-tasks/scripts",
        False,
        [],
        [
            partial(conditions.should_have_num_steps, 2),
            partial(conditions.should_have_catalog_execution_logs),
        ],
    ),
    (
        "01-tasks/stub",
        False,
        [],
        [
            partial(conditions.should_have_num_steps, 4),
            partial(conditions.should_have_catalog_execution_logs),
        ],
    ),
    (
        "01-tasks/notebook",
        False,
        [],
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
    # ("03-parameters/passing_parameters_notebook", False, []),
    # ("03-parameters/passing_parameters_python", False, []),
    # ("03-parameters/passing_parameters_shell", False, []),
    # ("03-parameters/static_parameters_non_python", False, []),
    # ("04-catalog/catalog", False, [mocked_context]),
    # ("06-parallel/parallel", False, []),
    # ("06-parallel/nesting", False, []),
    # ("07-map/map", False, []),
    # ("07-map/custom_reducer", False, []),
]


@pytest.mark.parametrize("example", list_python_examples())
@pytest.mark.parametrize("context", contexts)
# @pytest.mark.no_cover
@pytest.mark.e2e
def test_python_examples(example, context, monkeypatch, mocker):
    print(f"Testing {example}...")

    mod, fails, ignore_contexts, assertions = example
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
    file, fails, ignore_contexts, assertions = example

    if context in ignore_contexts:
        return

    context = context()
    example_file = f"examples/{file}.yaml"
    parameters_file = "examples/common/initial_parameters.yaml"

    with context:
        from runnable import exceptions
        from runnable.entrypoints import execute_pipeline_yaml_spec

        print("existing", os.environ.get(defaults.ENV_RUN_ID))
        run_id = generate_run_id()
        os.environ[defaults.ENV_RUN_ID] = run_id
        print(run_id)
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
