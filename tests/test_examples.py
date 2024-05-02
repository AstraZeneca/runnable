import importlib
import os
from contextlib import contextmanager

import pytest

from runnable import exceptions
from runnable.entrypoints import execute

# # (file, is_fail?)
python_examples = [
    ("01-tasks/notebook", False),
    ("01-tasks/python_tasks", False),
    ("01-tasks/scripts", False),
    ("01-tasks/stub", False),
    ("02-sequential/default_fail", True),
    ("02-sequential/on_failure_fail", True),
    ("02-sequential/on_failure_succeed", False),
    ("02-sequential/traversal", False),
    ("03-parameters/passing_parameters_notebook", False),
    ("03-parameters/passing_parameters_python", False),
    ("03-parameters/passing_parameters_shell", False),
    ("03-parameters/static_parameters_non_python", False),
    ("03-parameters/static_parameters_python", False),
    ("04-catalog/catalog", False),
    ("06-parallel/parallel", False),
    ("06-parallel/nesting", False),
    ("07-map/map", False),
    ("07-map/custom_reducer", False),
]


def list_python_examples():
    for example in python_examples:
        yield example


@contextmanager
def chunked_fs_context():
    os.environ["RUNNABLE_CONFIGURATION_FILE"] = "examples/configs/chunked-fs-run_log.yaml"
    os.environ["RUNNABLE_PRM_envvar"] = "from env"
    yield
    del os.environ["RUNNABLE_CONFIGURATION_FILE"]
    del os.environ["RUNNABLE_PRM_envvar"]


@contextmanager
def default_context():
    os.environ["RUNNABLE_PRM_envvar"] = "from env"
    yield
    del os.environ["RUNNABLE_PRM_envvar"]


contexts = [default_context, chunked_fs_context]


@pytest.mark.parametrize("example", list_python_examples())
@pytest.mark.parametrize("context", contexts)
# @pytest.mark.no_cover
@pytest.mark.e2e
def test_python_examples(example, context):
    print(f"Testing {example}...")

    mod, status = example
    context = context()

    imported_module = importlib.import_module(f"examples.{mod.replace('/', '.')}")
    f = getattr(imported_module, "main")
    with context:
        try:
            f()
        except exceptions.ExecutionFailedError:
            print("Example failed")
            if not status:
                raise


@pytest.mark.parametrize("example", list_python_examples())
@pytest.mark.parametrize("context", contexts)
# @pytest.mark.no_cover
@pytest.mark.e2e
def test_yaml_examples(example, context):
    print(f"Testing {example}...")
    file, status = example
    context = context()
    example_file = f"examples/{file}.yaml"
    parameters_file = "examples/common/initial_parameters.yaml"

    with context:
        try:
            execute(pipeline_file=example_file, parameters_file=parameters_file)
        except exceptions.ExecutionFailedError:
            if not status:
                raise


# TODO: Need to test argo and local container
