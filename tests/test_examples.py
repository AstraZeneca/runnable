import pytest
from pathlib import Path
import importlib
import logging

from magnus.entrypoints import execute
from magnus import exceptions

# (file, is_fail?, kwargs)
examples = [
    ("concepts/catalog.yaml", False, {"configuration_file": "examples/configs/fs-catalog.yaml"}),
    ("concepts/notebook_api_parameters.yaml", False, {"parameters_file": "examples/concepts/parameters.yaml"}),
    ("concepts/notebook_env_parameters.yaml", False, {"parameters_file": "examples/concepts/parameters.yaml"}),
    ("concepts/notebook_native_parameters.yaml", False, {"parameters_file": "examples/concepts/parameters.yaml"}),
    ("concepts/parallel.yaml", False, {}),
    ("concepts/simple_notebook.yaml", False, {}),
    ("concepts/simple.yaml", False, {}),
    ("concepts/task_shell_parameters.yaml", False, {"parameters_file": "examples/parameters_initial.yaml"}),
    ("concepts/task_shell_simple.yaml", False, {}),
    ("concepts/traversal.yaml", False, {}),
    ("catalog.yaml", False, {"configuration_file": "examples/configs/fs-catalog.yaml"}),
    ("contrived.yaml", False, {}),
    ("default-fail.yaml", True, {}),
    ("logging.yaml", False, {}),
    ("mocking.yaml", False, {}),
    ("on-failure.yaml", False, {}),
    ("parallel-fail.yaml", True, {}),
    ("parameters_env.yaml", False, {"parameters_file": "examples/parameters_initial.yaml"}),
    ("parameters_flow.yaml", False, {"parameters_file": "examples/parameters_initial.yaml"}),
    ("python-tasks.yaml", False, {"parameters_file": "examples/parameters_initial.yaml"}),
    ("retry-fail.yaml", True, {"configuration_file": "examples/configs/fs-catalog-run_log.yaml"}),
    ("retry-fixed.yaml", False, {"configuration_file": "examples/configs/fs-catalog-run_log.yaml"}),
]


def list_examples():
    for example in examples:
        yield example


@pytest.mark.parametrize("example", list_examples())
@pytest.mark.no_cover
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


python_examples = [
    ("catalog_api", False),
    ("catalog", False),
    ("contrived", False),
    ("mocking", False),
    ("on_failure", False),
    ("parameters_api", False),
    ("parameters", False),
    ("python-tasks", False),
    ("secrets", False),
    ("concepts.catalog", False),
    ("concepts.parallel", False),
    ("concepts.simple", False),
    ("concepts.task_api_parameters", False),
    ("concepts.task_env_parameters", False),
    ("concepts.task_native_parameters", False),
    ("concepts.traversal", False),
]


def list_python_examples():
    for example in python_examples:
        yield example


@pytest.mark.parametrize("example", list_python_examples())
@pytest.mark.no_cover
def test_python_examples(example):
    print(f"Testing {example}...")

    mod, status = example

    imported_module = importlib.import_module(f"examples.{mod}")
    f = getattr(imported_module, "main")
    try:
        f()
    except exceptions.ExecutionFailedError:
        if not status:
            raise
