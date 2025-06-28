import os
from typing import Any, Dict, Optional

import pytest
from pydantic import BaseModel, Field

from runnable.datastore import JsonParameter
from runnable.parameters import (
    bind_args_for_pydantic_model,
    filter_arguments_for_func,
    get_user_set_parameters,
)


# Test Models
class SimpleModel(BaseModel):
    name: str
    value: int


class ComplexModel(BaseModel):
    required: str
    optional: Optional[str] = None
    nested: SimpleModel


class ConfigModel(BaseModel):
    name: str = Field(default="default")
    settings: Dict[str, Any] = Field(default_factory=dict)


# Test Functions
def func_with_primitives(a: int, b: str, c: float = 0.0):
    return a, b, c


def func_with_model(model: SimpleModel, name: str = "default"):
    return model, name


def func_with_kwargs(name: str, **kwargs):
    return name, kwargs


@pytest.fixture
def clean_env():
    """Clean environment variables before and after tests"""
    original_env = dict(os.environ)

    # Clean env vars
    for key in list(os.environ.keys()):
        if key.startswith("RUNNABLE_PRM_"):
            del os.environ[key]

    yield

    # Restore original env vars
    os.environ.clear()
    os.environ.update(original_env)


# Tests for get_user_set_parameters
def test_get_parameters_basic(clean_env):
    """Test basic parameter retrieval"""
    os.environ["RUNNABLE_PRM_STR"] = '"test"'
    os.environ["RUNNABLE_PRM_INT"] = "42"
    os.environ["RUNNABLE_PRM_BOOL"] = "true"

    params = get_user_set_parameters()

    assert "str" in params
    assert params["str"].get_value() == "test"
    assert params["int"].get_value() == 42
    assert params["bool"].get_value() is True


def test_get_parameters_complex_json(clean_env):
    """Test parameter retrieval with complex JSON"""
    os.environ["RUNNABLE_PRM_DICT"] = '{"key": "value", "nested": {"num": 42}}'
    os.environ["RUNNABLE_PRM_LIST"] = "[1, 2, 3]"

    params = get_user_set_parameters()

    assert params["dict"].get_value() == {"key": "value", "nested": {"num": 42}}
    assert params["list"].get_value() == [1, 2, 3]


def test_get_parameters_invalid_json(clean_env):
    """Test handling of invalid JSON"""
    os.environ["RUNNABLE_PRM_INVALID"] = "not json"

    params = get_user_set_parameters()
    assert params["invalid"].get_value() == "not json"


def test_get_parameters_with_removal(clean_env):
    """Test parameter retrieval with removal"""
    os.environ["RUNNABLE_PRM_TEST"] = '"value"'

    params = get_user_set_parameters(remove=True)
    assert "test" in params
    assert "RUNNABLE_PRM_TEST" not in os.environ


# Tests for filter_arguments_for_func
def test_filter_args_primitives():
    """Test filtering primitive type arguments"""
    params = {
        "a": JsonParameter(kind="json", value=42),
        "b": JsonParameter(kind="json", value="test"),
        "extra": JsonParameter(kind="json", value="ignored"),
    }

    filtered = filter_arguments_for_func(func_with_primitives, params)
    assert filtered == {"a": 42, "b": "test"}
    assert "extra" not in filtered


def test_filter_args_with_model():
    """Test filtering with Pydantic model"""
    model_data = {"name": "test", "value": 42}
    params = {
        "model": JsonParameter(kind="json", value=model_data),
        "name": JsonParameter(kind="json", value="custom"),
    }

    filtered = filter_arguments_for_func(func_with_model, params)
    assert isinstance(filtered["model"], SimpleModel)
    assert filtered["model"].name == "test"
    assert filtered["model"].value == 42
    assert filtered["name"] == "custom"


def test_filter_args_with_kwargs():
    """Test filtering with kwargs"""
    params = {
        "name": JsonParameter(kind="json", value="test"),
        "extra1": JsonParameter(kind="json", value=1),
        "extra2": JsonParameter(kind="json", value="extra"),
    }

    filtered = filter_arguments_for_func(func_with_kwargs, params)
    assert filtered["name"] == "test"
    assert filtered["extra1"] == 1
    assert filtered["extra2"] == "extra"


def test_filter_args_missing_required():
    """Test handling of missing required arguments"""
    params = {"b": JsonParameter(kind="json", value="test")}

    with pytest.raises(ValueError) as exc_info:
        filter_arguments_for_func(func_with_primitives, params)
    assert "Parameter a is required" in str(exc_info.value)


# Tests for bind_args_for_pydantic_model
def test_bind_args_simple_model():
    """Test binding arguments to simple model"""
    params = {"name": "test", "value": 42}

    model = bind_args_for_pydantic_model(params, SimpleModel)
    assert isinstance(model, SimpleModel)
    assert model.name == "test"
    assert model.value == 42


def test_bind_args_complex_model():
    """Test binding arguments to complex model"""
    params = {"required": "test", "nested": {"name": "nested", "value": 42}}

    model = bind_args_for_pydantic_model(params, ComplexModel)
    assert isinstance(model, ComplexModel)
    assert model.required == "test"
    assert model.optional is None
    assert isinstance(model.nested, SimpleModel)
    assert model.nested.name == "nested"


def test_bind_args_with_extra_fields():
    """Test binding with extra fields that should be ignored"""
    params = {"name": "test", "extra": "ignored", "settings": {"key": "value"}}

    model = bind_args_for_pydantic_model(params, ConfigModel)
    assert model.name == "test"
    assert model.settings == {"key": "value"}
    assert not hasattr(model, "extra")
