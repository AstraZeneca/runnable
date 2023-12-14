import pytest
from pathlib import Path
import importlib

from magnus.entrypoints import execute
from magnus import exceptions

# (file, is_fail?, kwargs)
examples = [
    ("contrived.yaml", False, {}),
    ("default-fail.yaml", True, {}),
    ("logging.yaml", False, {}),
    ("mocking.yaml", False, {}),
    ("on-failure.yaml", False, {}),
    ("parallel.yaml", False, {}),
    ("parallel-fail.yaml", True, {}),
    ("parameters_flow.yaml", False, {"parameters_file": "examples/parameters_initial.yaml"}),
    ("python-tasks.yaml", False, {"parameters_file": "examples/parameters_initial.yaml"}),
    ("catalog.yaml", False, {"configuration_file": "examples/configs/fs-catalog.yaml"}),
    ("parameters_env.yaml", False, {"parameters_file": "examples/parameters_initial.yaml"}),
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
    ("contrived", False),
    ("mocking", False),
    ("parameters_api", False),
    ("parameters", False),
    ("python-tasks", False),
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
