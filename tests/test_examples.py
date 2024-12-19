import importlib
import os
import subprocess
from contextlib import contextmanager

import pytest

from runnable import exceptions
from runnable.entrypoints import execute


def list_python_examples():
    for example in python_examples:
        yield example


@contextmanager
def container_context():
    os.environ["RUNNABLE_CONFIGURATION_FILE"] = "examples/configs/local-container.yaml"
    os.environ["RUNNABLE_PRM_envvar"] = "from env"
    yield
    del os.environ["RUNNABLE_CONFIGURATION_FILE"]
    del os.environ["RUNNABLE_PRM_envvar"]


@contextmanager
def chunked_fs_context():
    os.environ["RUNNABLE_CONFIGURATION_FILE"] = (
        "examples/configs/chunked-fs-run_log.yaml"
    )
    os.environ["RUNNABLE_PRM_envvar"] = "from env"
    yield
    del os.environ["RUNNABLE_CONFIGURATION_FILE"]
    del os.environ["RUNNABLE_PRM_envvar"]


@contextmanager
def mocked_context():
    os.environ["RUNNABLE_CONFIGURATION_FILE"] = "examples/08-mocking/default.yaml"
    os.environ["RUNNABLE_PRM_envvar"] = "from env"
    yield
    del os.environ["RUNNABLE_CONFIGURATION_FILE"]
    del os.environ["RUNNABLE_PRM_envvar"]


@contextmanager
def patched_context():
    os.environ["RUNNABLE_CONFIGURATION_FILE"] = "examples/08-mocking/patching.yaml"
    os.environ["RUNNABLE_PRM_envvar"] = "from env"
    yield
    del os.environ["RUNNABLE_CONFIGURATION_FILE"]
    del os.environ["RUNNABLE_PRM_envvar"]


@contextmanager
def default_context():
    os.environ["RUNNABLE_PRM_envvar"] = "from env"
    yield
    del os.environ["RUNNABLE_PRM_envvar"]


@contextmanager
def argo_context():
    os.environ["RUNNABLE_CONFIGURATION_FILE"] = "examples/configs/argo-config.yaml"
    yield
    subprocess.run(["argo", "lint", "--offline", "argo-pipeline.yaml"], check=True)
    del os.environ["RUNNABLE_CONFIGURATION_FILE"]


contexts = [default_context, chunked_fs_context, mocked_context, argo_context]

python_examples = [
    ("01-tasks/notebook", False, []),
    ("01-tasks/python_tasks", False, []),
    ("01-tasks/scripts", False, []),
    ("01-tasks/stub", False, []),
    ("02-sequential/default_fail", True, []),
    # ("02-sequential/on_failure_fail", True, []), # need https://github.com/AstraZeneca/runnable/issues/156
    # ("02-sequential/on_failure_succeed", False, []), # https://github.com/AstraZeneca/runnable/issues/156
    ("02-sequential/traversal", False, []),
    ("03-parameters/passing_parameters_notebook", False, []),
    ("03-parameters/passing_parameters_python", False, []),
    ("03-parameters/passing_parameters_shell", False, []),
    ("03-parameters/static_parameters_non_python", False, []),
    ("03-parameters/static_parameters_python", False, []),
    ("04-catalog/catalog", False, [mocked_context]),
    ("06-parallel/parallel", False, []),
    ("06-parallel/nesting", False, []),
    ("07-map/map", False, []),
    ("07-map/custom_reducer", False, []),
]


@pytest.mark.parametrize("example", list_python_examples())
@pytest.mark.parametrize("context", contexts)
# @pytest.mark.no_cover
@pytest.mark.e2e
def test_python_examples(example, context):
    print(f"Testing {example}...")

    mod, status, ignore_contexts = example
    if context in ignore_contexts:
        return

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
    file, status, ignore_contexts = example

    if context in ignore_contexts:
        return

    context = context()
    example_file = f"examples/{file}.yaml"
    parameters_file = "examples/common/initial_parameters.yaml"

    with context:
        try:
            execute(pipeline_file=example_file, parameters_file=parameters_file)
        except exceptions.ExecutionFailedError:
            if not status:
                raise


@pytest.mark.parametrize("example", list_python_examples())
@pytest.mark.container
def test_python_examples_container(example):
    print(f"Testing {example}...")

    mod, status, _ = example
    context = container_context()

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
@pytest.mark.container
def test_yaml_examples_container(example):
    print(f"Testing {example}...")
    file, status, _ = example

    context = container_context()

    example_file = f"examples/{file}.yaml"
    parameters_file = "examples/common/initial_parameters.yaml"

    with context:
        try:
            execute(pipeline_file=example_file, parameters_file=parameters_file)
        except exceptions.ExecutionFailedError:
            if not status:
                raise
