from typing import Dict, Union

import pandas as pd
from pydantic import BaseModel


def hello():
    "The most basic function"
    print("Hello World!")


def raise_ex():
    "A function that raises an exception"
    raise Exception("This is an exception")


class ComplexParams(BaseModel):
    x: int
    foo: str


def read_initial_params_as_pydantic(
    integer: int,
    floater: float,
    stringer: str,
    pydantic_param: ComplexParams,
):
    assert integer == 1
    assert floater == 3.14
    assert stringer == "hello"
    assert pydantic_param.x == 10
    assert pydantic_param.foo == "bar"


def read_initial_params_as_json(
    integer: int,
    floater: float,
    stringer: str,
    pydantic_param: Dict[str, Union[int, str]],
):
    assert integer == 1
    assert floater == 3.14
    assert stringer == "hello"
    assert pydantic_param["x"] == 10
    assert pydantic_param["foo"] == "bar"


def write_parameter():
    integer = 1
    floater = 3.14
    c = ComplexParams(x=10, foo="bar")
    data = {"calories": [420, 380, 390], "duration": [50, 40, 45]}

    df = pd.DataFrame(data)
    score = 0.9

    return df, integer, floater, "hello", c, score


def read_parameter(
    df: pd.DataFrame,
    integer: int,
    floater: float,
    stringer: str,
    pydantic_param: ComplexParams,
    score: float,
):
    assert integer == 1
    assert floater == 3.14
    assert stringer == "hello"
    assert pydantic_param.x == 10
    assert pydantic_param.foo == "bar"
    assert df.shape == (3, 2)
    assert score == 0.9


def read_unpickled_parameter(
    integer: int,
    floater: float,
    stringer: str,
    pydantic_param: ComplexParams,
    score: float,
):
    assert integer == 1
    assert floater == 3.14
    assert stringer == "hello"
    assert pydantic_param.x == 10
    assert pydantic_param.foo == "bar"
    assert score == 0.9
