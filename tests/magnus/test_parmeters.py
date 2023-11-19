import os
import logging

import pytest

from pydantic import BaseModel, ValidationError
from dataclasses import dataclass

from magnus import defaults
from magnus.parameters import get_user_set_parameters, cast_parameters_as_type


def test_get_user_set_parameters_does_nothing_if_prefix_does_not_match(monkeypatch):
    monkeypatch.setenv("random", "value")

    assert get_user_set_parameters() == {}


def test_get_user_set_parameters_returns_the_parameter_if_prefix_match_int(monkeypatch):
    monkeypatch.setenv(defaults.PARAMETER_PREFIX + "key", "1")

    assert get_user_set_parameters() == {"key": 1}


def test_get_user_set_parameters_returns_the_parameter_if_prefix_match_string(monkeypatch):
    monkeypatch.setenv(defaults.PARAMETER_PREFIX + "key", '"value"')

    assert get_user_set_parameters() == {"key": "value"}


def test_get_user_set_parameters_removes_the_parameter_if_prefix_match_remove(monkeypatch):
    monkeypatch.setenv(defaults.PARAMETER_PREFIX + "key", "1")

    assert defaults.PARAMETER_PREFIX + "key" in os.environ

    get_user_set_parameters(remove=True)

    assert defaults.PARAMETER_PREFIX + "key" not in os.environ


def test_cast_parameters_as_type_with_pydantic_model():
    class MyModel(BaseModel):
        a: int
        b: str

    value = {"a": 1, "b": "test"}
    cast_value = cast_parameters_as_type(value, MyModel)

    assert isinstance(cast_value, MyModel)
    assert cast_value.a == 1
    assert cast_value.b == "test"


def test_cast_parameters_as_type_with_dataclass():
    @dataclass
    class MyDataClass:
        a: int
        b: str

    value = {"a": 1, "b": "test"}
    cast_value = cast_parameters_as_type(value, MyDataClass)

    assert isinstance(cast_value, MyDataClass)
    assert cast_value.a == 1
    assert cast_value.b == "test"


def test_cast_parameters_as_type_with_dict():
    value = {"a": 1, "b": "test"}
    cast_value = cast_parameters_as_type(value, dict)

    assert isinstance(cast_value, dict)
    assert cast_value == value


def test_cast_parameters_as_type_with_non_special_type():
    value = "1"
    cast_value = cast_parameters_as_type(value, int)

    assert isinstance(cast_value, int)
    assert cast_value == 1


def test_cast_parameters_as_type_with_none():
    value = None
    cast_value = cast_parameters_as_type(value, None)

    assert cast_value is None


def test_cast_parameters_as_type_with_invalid_value():
    class MyModel(BaseModel):
        a: int

    value = {"a": "test"}
    with pytest.raises(ValidationError):
        cast_parameters_as_type(value, MyModel)


def test_cast_parameters_as_type_with_invalid_type(caplog):
    value = "test"
    with caplog.at_level(logging.WARNING):
        cast_parameters_as_type(value, list)

    assert f"Casting {value} of {type(value)} to {list} seems wrong!!" in caplog.text
