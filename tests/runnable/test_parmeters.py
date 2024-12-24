import os

import pytest

from pydantic import BaseModel, ValidationError


from runnable import defaults
from runnable.datastore import JsonParameter
from runnable.parameters import (
    get_user_set_parameters,
    bind_args_for_pydantic_model,
    filter_arguments_for_func,
)


def test_get_user_set_parameters_does_nothing_if_prefix_does_not_match(monkeypatch):
    monkeypatch.setenv("random", "value")

    assert get_user_set_parameters() == {}


def test_get_user_set_parameters_removes_the_parameter_if_prefix_match_remove(
    monkeypatch,
):
    monkeypatch.setenv(defaults.PARAMETER_PREFIX + "key", "1")

    assert defaults.PARAMETER_PREFIX + "key" in os.environ

    get_user_set_parameters(remove=True)

    assert defaults.PARAMETER_PREFIX + "key" not in os.environ


def test_bind_args_for_pydantic_model_with_correct_params():
    class MyModel(BaseModel):
        a: int
        b: str

    params = {"a": 1, "b": "test"}
    bound_model = bind_args_for_pydantic_model(params, MyModel)

    assert isinstance(bound_model, MyModel)
    assert bound_model.a == 1
    assert bound_model.b == "test"


def test_bind_args_for_pydantic_model_with_extra_params():
    class MyModel(BaseModel):
        a: int
        b: str

    params = {"a": 1, "b": "test", "c": 2}
    bound_model = bind_args_for_pydantic_model(params, MyModel)

    assert isinstance(bound_model, MyModel)
    assert bound_model.a == 1
    assert bound_model.b == "test"


def test_bind_args_for_pydantic_model_with_missing_params():
    class MyModel(BaseModel):
        a: int
        b: str

    params = {"a": 1}
    with pytest.raises(ValidationError):
        bind_args_for_pydantic_model(params, MyModel)


def test_filter_arguments_for_func_with_simple_arguments():
    def func(a: int, b: str):
        pass

    params = {
        "a": JsonParameter(kind="json", value=1),
        "b": JsonParameter(kind="json", value="test"),
    }
    bound_args = filter_arguments_for_func(func, params)

    assert bound_args == {"a": 1, "b": "test"}


# def test_filter_arguments_for_func_with_pydantic_model_arguments():
#     class MyModel(BaseModel):
#         a: int
#         b: str

#     def func(inner: MyModel, c: str):
#         pass

#     params = {
#         "inner": {"a": JsonParameter(kind="json", value=1), "b": JsonParameter(kind="json", value="test")},
#         "c": JsonParameter(kind="json", value="test"),
#     }
#     bound_args = filter_arguments_for_func(func, params)

#     assert bound_args == {"inner": MyModel(a=1, b="test"), "c": "test"}


def test_filter_arguments_for_func_with_missing_arguments_but_defaults_present():
    def func(inner: int, c: str = "test"):
        pass

    params = {"inner": JsonParameter(kind="json", value=1)}
    bound_args = filter_arguments_for_func(func, params)

    assert bound_args == {"inner": 1}


def test_filter_arguments_for_func_with_missing_arguments_and_no_defaults():
    def func(inner: int, c: str):
        pass

    params = {"inner": JsonParameter(kind="json", value=1)}
    with pytest.raises(
        ValueError, match=r"Parameter c is required for func but not provided"
    ):
        _ = filter_arguments_for_func(func, params)


def test_filter_arguments_for_func_with_map_variable_sent_in():
    params = {"inner": JsonParameter(kind="json", value=1)}

    def func(inner: int, first: int, second: str):
        pass

    bound_args = filter_arguments_for_func(
        func, params, map_variable={"first": 1, "second": "test"}
    )
    assert bound_args == {"inner": 1, "first": 1, "second": "test"}
