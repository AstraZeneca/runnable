import pytest

from magnus.extensions import nodes
from magnus import sdk


def test_success_init():
    test_success = sdk.Success()

    assert test_success.name == "success"
    assert test_success._node == nodes.SuccessNode(name="success", internal_name="")
