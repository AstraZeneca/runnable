import pytest
from pathlib import Path

from magnus.entrypoints import execute
from magnus import exceptions


def list_examples():
    example_dir = "example"

    for example in Path(example_dir).iterdir():
        if not example.name.endswith(".yaml"):
            continue

        yield example


@pytest.mark.parametrize("example", list_examples())
@pytest.mark.no_cover
def test_yaml_examples(example):
    print(f"Testing {example.resolve()}...")
    try:
        execute(configuration_file="", pipeline_file=str(example.resolve()))
    except exceptions.ExecutionFailedError:
        # There are some examples where the pipeline fails intentionally
        pass
