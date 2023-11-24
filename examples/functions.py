import logging

from pydantic import BaseModel

# Magnus logging levels are different to your logging levels.
logger = logging.getLogger("application")
logger.setLevel(logging.DEBUG)


class Parameter(BaseModel):
    x: int


def return_parameter() -> Parameter:
    # Return type of a function should be a pydantic model
    return Parameter(x=1)


def get_parameter(x: int):
    # Input args can be a pydantic model or the individual attributes of the model.
    logger.info(f"I got a parameter: {x}")
    print(x)
