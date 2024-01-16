from typing import List

from pydantic import create_model


def generate_array():
    return create_model("DynamicModel", array=(List[int], [1, 2, 3, 4, 5]))()


def execute_every_element(i: int):
    print(i)
