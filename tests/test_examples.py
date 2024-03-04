from contextlib import nullcontext, contextmanager
import pytest
from pathlib import Path
import os
import importlib
import subprocess

from runnable.entrypoints import execute
from runnable import exceptions

# (file, is_fail?, kwargs)
examples = [
    ("concepts/catalog.yaml", False, {"configuration_file": "examples/configs/fs-catalog.yaml"}),
    ("concepts/map.yaml", False, {}),
    ("concepts/map_shell.yaml", False, {}),
    ("concepts/nesting.yaml", False, {}),
    ("concepts/notebook_native_parameters.yaml", False, {"parameters_file": "examples/concepts/parameters.yaml"}),
    ("concepts/parallel.yaml", False, {}),
    ("concepts/simple_notebook.yaml", False, {}),
    ("concepts/simple.yaml", False, {}),
    ("concepts/traversal.yaml", False, {}),
    ("catalog.yaml", False, {"configuration_file": "examples/configs/fs-catalog.yaml"}),
    ("default-fail.yaml", True, {}),
    ("logging.yaml", False, {}),
    ("mocking.yaml", False, {}),
    ("on-failure.yaml", False, {}),
    ("parallel-fail.yaml", True, {}),
    ("parameters_flow.yaml", False, {"parameters_file": "examples/parameters_initial.yaml"}),
    ("python-tasks.yaml", False, {"parameters_file": "examples/parameters_initial.yaml"}),
]


def list_examples():
    for example in examples:
        yield example


@pytest.mark.parametrize("example", list_examples())
@pytest.mark.no_cover
@pytest.mark.e2e
def test_yaml_examples(example):
    print(f"Testing {example}...")
    examples_path = Path("examples")
    file_path, status, kwargs = example
    try:
        full_file_path = examples_path / file_path
        configuration_file = kwargs.pop("configuration_file", "")
        execute(configuration_file=configuration_file, pipeline_file=str(full_file_path.resolve()), **kwargs)
    except exceptions.ExecutionFailedError:
        if not status:
            raise


@pytest.mark.parametrize("example", list_examples())
@pytest.mark.no_cover
@pytest.mark.e2e
def test_yaml_examples_argo(example):
    print(f"Testing {example}...")
    examples_path = Path("examples")
    file_path, status, kwargs = example
    try:
        full_file_path = examples_path / file_path
        kwargs.pop("configuration_file", "")
        configuration_file = "examples/configs/argo-config.yaml"
        execute(configuration_file=configuration_file, pipeline_file=str(full_file_path.resolve()), **kwargs)
        subprocess.run(["argo", "lint", "--offline", "argo-pipeline.yaml"], check=True)
    except exceptions.ExecutionFailedError:
        if not status:
            raise


@pytest.mark.parametrize("example", list_examples())
@pytest.mark.no_cover
@pytest.mark.e2e_container
def test_yaml_examples_container(example):
    print(f"Testing {example}...")
    examples_path = Path("examples")
    file_path, status, kwargs = example
    try:
        full_file_path = examples_path / file_path
        kwargs.pop("configuration_file", "")
        configuration_file = "examples/configs/local-container.yaml"
        os.environ["runnable_VAR_default_docker_image"] = "runnable:3.8"
        execute(configuration_file=configuration_file, pipeline_file=str(full_file_path), **kwargs)
    except exceptions.ExecutionFailedError:
        if not status:
            raise


@contextmanager
def secrets_env_context():
    os.environ["secret"] = "secret_value"
    os.environ["runnable_CONFIGURATION_FILE"] = "examples/configs/secrets-env-default.yaml"
    yield
    del os.environ["secret"]
    del os.environ["runnable_CONFIGURATION_FILE"]


# function, success, context
python_examples = [
    ("catalog", False, None),
    ("catalog_api", False, None),
    ("catalog_simple", False, None),
    ("mocking", False, None),
    ("on_failure", False, None),
    ("parameters", False, None),
    ("python-tasks", False, None),
    ("secrets", False, None),
    ("concepts.catalog", False, None),
    ("concepts.catalog_api", False, None),
    ("concepts.catalog_object", False, None),
    ("concepts.map", False, None),
    ("concepts.nesting", False, None),
    ("concepts.parallel", False, None),
    ("concepts.simple", False, None),
    ("concepts.task_native_parameters", False, None),
    ("concepts.traversal", False, None),
]


def list_python_examples():
    for example in python_examples:
        yield example


@pytest.mark.parametrize("example", list_python_examples())
@pytest.mark.no_cover
@pytest.mark.e2e
def test_python_examples(example):
    print(f"Testing {example}...")

    mod, status, context = example

    if not context:
        context = nullcontext()
    else:
        context = context()

    imported_module = importlib.import_module(f"examples.{mod}")
    f = getattr(imported_module, "main")
    try:
        with context:
            f()
    except exceptions.ExecutionFailedError:
        if not status:
            raise
