from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from pydantic import BaseModel


def hello():
    "The most basic function"
    print("Hello World!")


def mocked_hello():
    "Mock of the hello function"
    print("Hello from mock")


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
    envvar: str,
):
    assert integer == 1
    assert floater == 3.14
    assert stringer == "hello"
    assert pydantic_param.x == 10
    assert pydantic_param.foo == "bar"
    assert envvar == "from env"


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


def write_files():
    data = {"calories": [420, 380, 390], "duration": [50, 40, 45]}
    df = pd.DataFrame(data)

    df.to_csv("df.csv", index=False)

    Path("data_folder").mkdir(parents=True, exist_ok=True)
    with open("data_folder/data.txt", "w", encoding="utf-8") as f:
        f.write("hello world")


def read_files():
    df = pd.read_csv("df.csv")
    assert df.shape == (3, 2)

    with open("data_folder/data.txt", "r", encoding="utf-8") as f:
        data = f.read()

    assert data.strip() == "hello world"


def process_chunk(chunk: int):
    """
    An example function that processes a chunk of data.
    We are multiplying the chunk by 10.
    """
    return chunk * 10


def read_processed_chunk(
    chunk: int, processed_python: int, processed_notebook: int, processed_shell: int
):
    """
    A downstream step of process_chunk of map state which reads the processed chunk.
    Since the process_chunk returns the chunk multiplied by 10, we assert that.
    """
    assert int(chunk) * 10 == processed_python
    assert processed_python * 10 == processed_notebook
    assert processed_notebook * 10 == processed_shell


def assert_default_reducer(
    processed_python: List[int],
    processed_notebook: List[int],
    processed_shell: List[int],
    chunks: List[int],
):
    """
    Demonstrates the default reducer which just returns the list of processed chunks.
    """
    assert processed_python == [chunk * 10 for chunk in chunks]
    assert processed_notebook == [chunk * 100 for chunk in chunks]
    assert processed_shell == [chunk * 1000 for chunk in chunks]


def assert_custom_reducer(
    processed_python: int,
    processed_notebook: int,
    processed_shell: int,
    chunks: List[int],
):
    """
    Asserts the custom reducer returns the max of all the processed chunks.
    """
    assert processed_python == max(chunk * 10 for chunk in chunks)
    assert processed_notebook == max(chunk * 100 for chunk in chunks)
    assert processed_shell == max(chunk * 1000 for chunk in chunks)
